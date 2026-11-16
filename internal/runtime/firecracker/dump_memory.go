package firecracker

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"k8s.io/klog/v2"

	runtimecontract "fast-sandbox/internal/runtime/contract"
)

// Memory headroom for the full-snapshot memory dump.
//
// Firecracker writes the whole guest memory to the dump file with buffered
// I/O from its own process, so the page cache of that file is charged to the
// memory cgroup that caps the VM. That cgroup normally sits at its limit
// (page cache is only reclaimed on demand), and the kernel lets a share of
// the cgroup's file memory (vm.dirty_ratio) become dirty before it throttles
// the writer: gigabytes for a 16 GiB guest. When most of the cgroup is memory
// reclaim cannot free quickly - a guest whose memory is full (anonymous
// pages), or a guest restored from a checkpoint, whose memory is file pages
// mapped by the VM and just touched by the dump itself - the dump's dirty
// backlog does not fit, the write-path allocation cannot wait for its own
// writeback, and the cgroup OOM-kills the VM (and, in a legacy pool, the
// Fastlet that shares its cgroup).
//
// For the duration of a full snapshot the driver lowers memory.high of that
// cgroup to its limit minus a reserve. Above memory.high the kernel reclaims
// and throttles the allocating task instead of invoking the OOM killer, which
// keeps the backlog to what writeback drains. The line is never set below the
// memory reclaim cannot free (anonymous, shared and kernel memory): a
// memory.high write below the usage reclaims synchronously and, where the
// kernel ignores O_NONBLOCK on it, does not return while the memory cannot be
// reclaimed. If that memory reaches the upper half of the reserve, the
// snapshot is refused with ErrInsufficientMemory and the VM is never paused.

const (
	// dumpMemoryReserveFloor is the smallest gap kept between memory.high and
	// memory.max while a dump runs.
	dumpMemoryReserveFloor int64 = 256 << 20
	// dumpMemoryReserveShift sizes the reserve as limit>>5 (about 3%): the
	// kernel lets usage run roughly 1-2% past memory.high before its
	// throttling delay becomes significant, so the reserve must cover that
	// overshoot on large cgroups.
	dumpMemoryReserveShift = 5
	// cgroupFSRoot is where the Fastlet sees the host's cgroup v2 tree.
	cgroupFSRoot = "/sys/fs/cgroup"
)

// dumpMemoryReserve returns the gap between memory.max and the memory.high
// applied during a dump.
func dumpMemoryReserve(limit int64) int64 {
	return max(dumpMemoryReserveFloor, limit>>dumpMemoryReserveShift)
}

// dumpMemoryHolds counts the in-flight snapshots per cgroup (several VMs of
// a Fastlet can share one), so memory.high is lowered by the first and
// restored by the last.
var dumpMemoryHolds = struct {
	sync.Mutex
	byPath map[string]*dumpMemoryHold
}{byPath: map[string]*dumpMemoryHold{}}

type dumpMemoryHold struct {
	count    int
	previous string
}

// reserveSnapshotDumpMemory applies the dump memory reserve for the plan's
// Sandbox. A state that cannot be read leaves the dump to report the precise
// error.
func (d *Driver) reserveSnapshotDumpMemory(plan dumpPlan) (func(), error) {
	state, err := loadState(plan.sandboxDir)
	if err != nil {
		return func() {}, nil //nolint:nilerr // the dump itself reports an unreadable state
	}
	return d.reserveDumpMemory(plan.sandboxID, state.PID)
}

// reserveDumpMemory lowers memory.high of the cgroup that caps the VM with
// the given process ID and returns the function that restores it. It returns
// a no-op when the VM's memory is not capped (or the cgroup cannot be found:
// the dump then runs as before), and ErrInsufficientMemory when the memory
// reclaim cannot free reaches the upper half of the reserve.
func (d *Driver) reserveDumpMemory(sandboxID string, pid int) (func(), error) {
	noop := func() {}
	resolve := d.dumpMemoryCgroup
	if resolve == nil {
		resolve = d.vmMemoryCgroup
	}
	path, err := resolve(pid)
	if err != nil || path == "" {
		klog.InfoS("firecracker dump memory cgroup not found; dumping without a memory reserve",
			"sandboxID", sandboxID, "pid", pid, "err", err)
		return noop, nil
	}
	limit, capped, err := readCgroupLimit(path, "memory.max")
	if err != nil || !capped {
		return noop, nil //nolint:nilerr // an unreadable or unlimited cgroup leaves the dump as it was
	}
	high := limit - dumpMemoryReserve(limit)

	dumpMemoryHolds.Lock()
	defer dumpMemoryHolds.Unlock()
	hold := dumpMemoryHolds.byPath[path]
	if hold == nil {
		previous, err := os.ReadFile(filepath.Join(path, "memory.high"))
		if err != nil {
			return noop, nil //nolint:nilerr // no memory.high control: nothing to lower
		}
		hold = &dumpMemoryHold{previous: strings.TrimSpace(string(previous))}
		pinned, err := unreclaimableMemory(path)
		if err != nil {
			klog.InfoS("firecracker dump memory stat unreadable; dumping without a memory reserve",
				"sandboxID", sandboxID, "cgroup", path, "err", err)
			return noop, nil
		}
		if pinned > high+(limit-high)/2 {
			return nil, fmt.Errorf("%w: the VM's memory cgroup holds %d bytes that cannot be reclaimed (limit %d); the memory dump needs %d bytes of headroom",
				runtimecontract.ErrInsufficientMemory, pinned, limit, limit-high)
		}
		line := max(high, pinned)
		// Never raise an existing, lower memory.high.
		if existing, ok := parseCgroupLimit(hold.previous); ok && existing <= line {
			line = existing
		} else if err := writeMemoryHigh(path, strconv.FormatInt(line, 10)); err != nil {
			klog.ErrorS(err, "firecracker dump memory reserve not applied", "sandboxID", sandboxID, "cgroup", path)
			return noop, nil
		}
		dumpMemoryHolds.byPath[path] = hold
		klog.InfoS("firecracker dump memory reserve applied",
			"sandboxID", sandboxID, "cgroup", path, "memoryMax", limit, "memoryHigh", line, "unreclaimable", pinned)
	}
	hold.count++
	var once sync.Once
	return func() {
		once.Do(func() {
			dumpMemoryHolds.Lock()
			defer dumpMemoryHolds.Unlock()
			hold.count--
			if hold.count > 0 {
				return
			}
			delete(dumpMemoryHolds.byPath, path)
			if err := writeMemoryHigh(path, hold.previous); err != nil {
				klog.ErrorS(err, "firecracker dump memory reserve not released", "cgroup", path, "memoryHigh", hold.previous)
			}
		})
	}, nil
}

// unreclaimableMemory sums the cgroup memory that reclaim cannot free
// without swap: anonymous and shared pages plus kernel memory.
func unreclaimableMemory(path string) (int64, error) {
	file, err := os.Open(filepath.Join(path, "memory.stat"))
	if err != nil {
		return 0, err
	}
	defer file.Close()
	stat := map[string]int64{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		key, value, ok := strings.Cut(scanner.Text(), " ")
		if !ok {
			continue
		}
		if parsed, err := strconv.ParseInt(value, 10, 64); err == nil {
			stat[key] = parsed
		}
	}
	if err := scanner.Err(); err != nil {
		return 0, err
	}
	if _, ok := stat["anon"]; !ok {
		return 0, errors.New("memory.stat carries no anon counter")
	}
	kernel, ok := stat["kernel"]
	if !ok {
		// Kernels before 5.18 report the kernel memory in parts.
		kernel = stat["kernel_stack"] + stat["pagetables"] + stat["percpu"] + stat["sock"] + stat["slab_unreclaimable"] + stat["vmalloc"]
	}
	return stat["anon"] + stat["shmem"] + kernel, nil
}

// writeMemoryHigh sets memory.high without waiting for reclaim where the
// kernel supports O_NONBLOCK on the control.
func writeMemoryHigh(path, value string) error {
	file, err := os.OpenFile(filepath.Join(path, "memory.high"), os.O_WRONLY|os.O_TRUNC|syscall.O_NONBLOCK, 0)
	if err != nil {
		return err
	}
	_, err = file.WriteString(value)
	return errors.Join(err, file.Close())
}

// vmMemoryCgroup returns the cgroup that caps the memory of the VM process:
// the bounded runtime subtree in ResourceAware mode, otherwise the nearest
// ancestor of the process's own cgroup with a finite memory.max (in a legacy
// pool, the Fastlet container's cgroup, which the VM shares).
func (d *Driver) vmMemoryCgroup(pid int) (string, error) {
	if d.runtimeCgroups != nil {
		return d.runtimeCgroups.AggregateFilesystemPath(), nil
	}
	return processMemoryCgroup("/proc", cgroupFSRoot, pid)
}

func processMemoryCgroup(procRoot, cgroupRoot string, pid int) (string, error) {
	if pid <= 0 {
		return "", errors.New("sandbox state carries no VM process ID")
	}
	data, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "cgroup"))
	if err != nil {
		return "", err
	}
	relative := ""
	for _, line := range strings.Split(string(data), "\n") {
		if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "0::"); ok {
			relative = filepath.Clean("/" + rest)
			break
		}
	}
	if relative == "" {
		return "", errors.New("VM process is not in a cgroup v2 hierarchy")
	}
	root := filepath.Clean(cgroupRoot)
	for path := filepath.Join(root, relative); path != root && strings.HasPrefix(path, root+string(filepath.Separator)); path = filepath.Dir(path) {
		if _, capped, err := readCgroupLimit(path, "memory.max"); err == nil && capped {
			return path, nil
		}
	}
	return "", nil
}

// readCgroupLimit reads a limit control; capped is false for "max".
func readCgroupLimit(path, name string) (int64, bool, error) {
	data, err := os.ReadFile(filepath.Join(path, name))
	if err != nil {
		return 0, false, err
	}
	value, capped := parseCgroupLimit(strings.TrimSpace(string(data)))
	return value, capped, nil
}

func parseCgroupLimit(raw string) (int64, bool) {
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || value <= 0 {
		return 0, false
	}
	return value, true
}
