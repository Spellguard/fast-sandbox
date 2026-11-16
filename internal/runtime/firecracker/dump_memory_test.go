package firecracker

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	runtimecontract "fast-sandbox/internal/runtime/contract"
)

// fakeMemoryCgroup writes the memory controls of a cgroup directory; anon
// is the only unreclaimable memory it reports.
func fakeMemoryCgroup(t *testing.T, dir, limit, high string, anon int64) string {
	t.Helper()
	require.NoError(t, os.MkdirAll(dir, 0o750))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "memory.max"), []byte(limit+"\n"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "memory.high"), []byte(high+"\n"), 0o600))
	stat := fmt.Sprintf("anon %d\nfile 1073741824\nkernel 0\nshmem 0\nfile_dirty 0\n", anon)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "memory.stat"), []byte(stat), 0o600))
	return dir
}

func readMemoryHigh(t *testing.T, dir string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, "memory.high"))
	require.NoError(t, err)
	return strings.TrimSpace(string(data))
}

func TestDumpMemoryReserve(t *testing.T) {
	require.Equal(t, dumpMemoryReserveFloor, dumpMemoryReserve(4<<30), "small cgroups keep the floor")
	require.Equal(t, int64(17152<<20)/32, dumpMemoryReserve(17152<<20))
	require.Equal(t, int64(19968<<20)/32, dumpMemoryReserve(19968<<20))
}

func TestUnreclaimableMemory(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "memory.stat"), []byte("anon 100\nfile 5000\nfile_mapped 4000\nkernel 20\nshmem 3\nslab_unreclaimable 7\n"), 0o600))
	pinned, err := unreclaimableMemory(dir)
	require.NoError(t, err)
	require.Equal(t, int64(123), pinned, "file pages, mapped or not, are reclaimable")

	// Kernels before 5.18 have no kernel counter.
	require.NoError(t, os.WriteFile(filepath.Join(dir, "memory.stat"), []byte("anon 100\nshmem 3\nkernel_stack 1\npagetables 2\npercpu 3\nsock 4\nslab_unreclaimable 5\nvmalloc 6\n"), 0o600))
	pinned, err = unreclaimableMemory(dir)
	require.NoError(t, err)
	require.Equal(t, int64(124), pinned)
}

func TestProcessMemoryCgroupFindsTheCappedAncestor(t *testing.T) {
	proc, root := t.TempDir(), t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(proc, "42"), 0o750))
	require.NoError(t, os.WriteFile(filepath.Join(proc, "42", "cgroup"), []byte("0::/kubepods/pod1/ctr1/leaf\n"), 0o600))
	fakeMemoryCgroup(t, filepath.Join(root, "kubepods/pod1/ctr1/leaf"), "max", "max", 0)
	capped := fakeMemoryCgroup(t, filepath.Join(root, "kubepods/pod1/ctr1"), "17985175552", "max", 0)
	fakeMemoryCgroup(t, filepath.Join(root, "kubepods/pod1"), "21474836480", "max", 0)

	path, err := processMemoryCgroup(proc, root, 42)
	require.NoError(t, err)
	require.Equal(t, capped, path, "the nearest finite memory.max caps the VM")

	require.NoError(t, os.WriteFile(filepath.Join(capped, "memory.max"), []byte("max\n"), 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(root, "kubepods/pod1/memory.max"), []byte("max\n"), 0o600))
	path, err = processMemoryCgroup(proc, root, 42)
	require.NoError(t, err)
	require.Empty(t, path, "an uncapped VM needs no reserve")

	_, err = processMemoryCgroup(proc, root, 0)
	require.Error(t, err)
}

func TestReserveDumpMemoryLowersHighUntilTheLastRelease(t *testing.T) {
	limit := int64(19968 << 20)
	dir := fakeMemoryCgroup(t, filepath.Join(t.TempDir(), "ctr"), strconv.FormatInt(limit, 10), "max", 10<<30)
	driver := &Driver{dumpMemoryCgroup: func(int) (string, error) { return dir, nil }}

	first, err := driver.reserveDumpMemory("sb-1", 100)
	require.NoError(t, err)
	want := strconv.FormatInt(limit-dumpMemoryReserve(limit), 10)
	require.Equal(t, want, readMemoryHigh(t, dir))

	// A second VM of the same cgroup dumps while the first still publishes.
	second, err := driver.reserveDumpMemory("sb-2", 101)
	require.NoError(t, err)
	first()
	first() // idempotent
	require.Equal(t, want, readMemoryHigh(t, dir), "held while another snapshot runs")
	second()
	require.Equal(t, "max", readMemoryHigh(t, dir), "the last release restores the previous value")
}

func TestReserveDumpMemoryRefusesWhenTheMemoryCannotBeReclaimed(t *testing.T) {
	limit := int64(17152 << 20)
	// Anonymous memory in the upper half of the reserve (for example a guest
	// whose memory is full in a cgroup with no headroom).
	dir := fakeMemoryCgroup(t, filepath.Join(t.TempDir(), "ctr"), strconv.FormatInt(limit, 10), "max", limit-dumpMemoryReserve(limit)/2+1)
	driver := &Driver{dumpMemoryCgroup: func(int) (string, error) { return dir, nil }}

	release, err := driver.reserveDumpMemory("sb-1", 100)
	require.ErrorIs(t, err, runtimecontract.ErrInsufficientMemory)
	require.Nil(t, release)
	require.Equal(t, "max", readMemoryHigh(t, dir), "a refused reserve leaves memory.high alone")
}

func TestReserveDumpMemoryNeverSetsTheLineBelowUnreclaimableMemory(t *testing.T) {
	limit := int64(17152 << 20)
	anon := limit - dumpMemoryReserve(limit) + (16 << 20)
	dir := fakeMemoryCgroup(t, filepath.Join(t.TempDir(), "ctr"), strconv.FormatInt(limit, 10), "max", anon)
	driver := &Driver{dumpMemoryCgroup: func(int) (string, error) { return dir, nil }}

	release, err := driver.reserveDumpMemory("sb-1", 100)
	require.NoError(t, err)
	require.Equal(t, strconv.FormatInt(anon, 10), readMemoryHigh(t, dir))
	release()
	require.Equal(t, "max", readMemoryHigh(t, dir))
}

func TestReserveDumpMemoryKeepsAnExistingLowerHigh(t *testing.T) {
	limit := int64(8 << 30)
	dir := fakeMemoryCgroup(t, filepath.Join(t.TempDir(), "ctr"), strconv.FormatInt(limit, 10), strconv.FormatInt(4<<30, 10), 1<<30)
	driver := &Driver{dumpMemoryCgroup: func(int) (string, error) { return dir, nil }}

	release, err := driver.reserveDumpMemory("sb-1", 100)
	require.NoError(t, err)
	require.Equal(t, strconv.FormatInt(4<<30, 10), readMemoryHigh(t, dir))
	release()
	require.Equal(t, strconv.FormatInt(4<<30, 10), readMemoryHigh(t, dir))
}

func TestReserveDumpMemoryIsANoOpWithoutALimit(t *testing.T) {
	dir := fakeMemoryCgroup(t, filepath.Join(t.TempDir(), "ctr"), "max", "max", 1<<30)
	driver := &Driver{dumpMemoryCgroup: func(int) (string, error) { return dir, nil }}
	release, err := driver.reserveDumpMemory("sb-1", 100)
	require.NoError(t, err)
	release()
	require.Equal(t, "max", readMemoryHigh(t, dir))

	driver.dumpMemoryCgroup = func(int) (string, error) { return "", nil }
	release, err = driver.reserveDumpMemory("sb-1", 100)
	require.NoError(t, err)
	release()
}

func TestCreateSnapshotRefusesBeforePauseWithoutDumpMemoryHeadroom(t *testing.T) {
	fixture, agent := newSnapshotFixture(t)
	limit := int64(17152 << 20)
	dir := fakeMemoryCgroup(t, filepath.Join(t.TempDir(), "ctr"), strconv.FormatInt(limit, 10), "max", limit)
	fixture.driver.dumpMemoryCgroup = func(int) (string, error) { return dir, nil }
	sandboxID := seedRunningSandbox(t, fixture, PhaseRunning)

	_, err := fixture.driver.CreateSnapshot(context.Background(), &runtimecontract.SnapshotInput{
		SandboxID: sandboxID, SnapshotID: "snap-1", TemplateName: "app-v2",
	})
	require.ErrorIs(t, err, runtimecontract.ErrInsufficientMemory)
	// The VM was never paused and nothing was published.
	require.True(t, fixture.server.running)
	require.Empty(t, fixture.server.snapshotDumps)
	require.Empty(t, agent.fakeAgentClient.publishes)
	require.Equal(t, "max", readMemoryHigh(t, dir))
}

func TestCreateSnapshotHoldsDumpMemoryReserveThroughPublication(t *testing.T) {
	fixture, agent := newSnapshotFixture(t)
	limit := int64(19968 << 20)
	dir := fakeMemoryCgroup(t, filepath.Join(t.TempDir(), "ctr"), strconv.FormatInt(limit, 10), "max", 1<<30)
	fixture.driver.dumpMemoryCgroup = func(int) (string, error) { return dir, nil }
	sandboxID := seedRunningSandbox(t, fixture, PhaseRunning)
	var highAtPublish string
	agent.onPublish = func() { highAtPublish = readMemoryHigh(t, dir) }

	_, err := fixture.driver.CreateSnapshot(context.Background(), &runtimecontract.SnapshotInput{
		SandboxID: sandboxID, SnapshotID: "snap-1", TemplateName: "app-v2",
	})
	require.NoError(t, err)
	require.Len(t, fixture.server.snapshotDumps, 1)
	require.Equal(t, strconv.FormatInt(limit-dumpMemoryReserve(limit), 10), highAtPublish)
	require.Equal(t, "max", readMemoryHigh(t, dir), "released once the snapshot is done")
}
