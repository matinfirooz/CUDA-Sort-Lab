# How the CUDA implementations work

The source is organized around synchronization boundaries. A kernel launch in
the same stream follows the preceding work in that stream. A block barrier
orders only threads in that block. The implementations never use a block
barrier as a grid-wide sorting-stage boundary.

## Bitonic network

Pad N keys to the next power of two P with `INT_MAX`. For each sequence size
2, 4, 8, ..., P, perform compare-exchange steps with XOR distances
size/2, size/4, ..., 1. The partner of thread i is `i ^ distance`.

Only the smaller index owns a pair and writes both elements. This prevents two
threads from updating the same pair. `(i & size) == 0` selects ascending order;
the other groups sort descending while building a bitonic sequence. At the
final size P, the complete sequence is ascending.

Each distance is a separate global kernel, or a block-wide barrier in the
fixed 256-key quicksort leaf. Padding is safe even when real keys equal
`INT_MAX`: only key values are returned, with the first N values retained.

## Merge through rank searches

Start with sorted runs of length one, then double the run length every pass.
One thread handles each element in a pair of sorted runs.

- A left-run key uses **lower_bound** in the right run: count right keys < key.
- A right-run key uses **upper_bound** in the left run: count left keys <= key.
- Add that count to the key's position within its own run and the pair's base.

For left `[-1, 3]` and right `[0, 3]`, the left 3 goes to position 2 and the
right 3 to position 3. Equal keys therefore have different destinations.
Using lower_bound on both sides would allow collisions between equal keys.

Input and output are separate buffers during a pass. Their roles swap after
each kernel, so writes cannot destroy values still being searched.
Binary searching per key adds a logarithmic factor: this implementation uses
O(N log² N) work. A merge-path implementation is a possible later optimization.

## Stable binary LSD radix

The ordering key is `uint32(value) ^ 0x80000000`. The sign-bit flip makes
unsigned ordering of transformed keys match signed ordering of the original
integers, including `INT_MIN` and `INT_MAX`. Original values are stored in
the output buffers; transformed keys are used only for classification.

Process bits 0 through 31. For each key, write flag 1 if the current bit is
zero. An exclusive scan gives `prefix[i]`, the number of zero-bit keys before i.
Let Z be the total number of zero-bit keys:

| Current bit | Destination |
| --- | --- |
| 0 | `prefix[i]` |
| 1 | `Z + i - prefix[i]` |

Both groups retain their previous order. That stability is essential to LSD
radix sorting: the order established by lower bits survives higher-bit passes.
After 32 passes, all signed 32-bit keys are sorted.

The custom scan loads 512 values per 256-thread block, pads the last block
with zeros, performs a Blelloch upsweep/downsweep, recursively scans block
totals, and adds the corresponding total prefix to every local output.
All block threads participate in every barrier, including padded lanes.

## Counting and bucket distribution

Counting sort accepts an explicit inclusive range. Histogram index arithmetic
uses 64 bits before the validated offset is narrowed to an int. An invalid
key sets an error flag instead of accessing outside the histogram.

For input `[-1, 3, 0, 3]`, the range is [-1,3]:

| Key | -1 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- |
| Count | 1 | 1 | 0 | 0 | 2 |
| Exclusive start | 0 | 1 | 2 | 2 | 2 |

Output position i finds the final start <= i with upper_bound. Position 2
must choose the final repeated start, corresponding to key 3. This skips
empty bins correctly. Histogram reconstruction uses O(N log(B+1)) work;
it trades extra searches for one independent thread per output key.

Bucket sort uses the top eight bits of the same sign-adjusted key. This
monotonic mapping gives 256 ordered buckets. Histogram + exclusive scan
creates each bucket's output interval. Atomic cursor increments allocate
unique scatter slots. Heap-sort each bucket with one thread, then concatenate
the already ordered intervals. The per-bucket heap work is serial, and narrow
input ranges can concentrate almost all work into very few buckets.

## Three-way quicksort

The host maintains two device work queues of disjoint half-open ranges.
Each kernel gives one block ownership of one range.

1. Choose the median of the first, middle, and last keys as pivot.
2. Count keys below, equal to, and above the pivot with shared atomics.
3. Form three output intervals from those counts.
4. Scatter into a separate scratch range using shared atomic cursors.
5. Synchronize the block, then copy the completed scratch range back.
6. Enqueue the below/above ranges if their lengths exceed one.

Ranges of at most 256 keys use a shared-memory bitonic network. The equal
region needs no further work, so an all-equal input terminates after one
partition (or one leaf sort). Because the pivot is an existing key, every
partition removes at least one key from further recursion.

Each queue entry owns at least two keys. Ranges are disjoint, so at most
floor(N/2) entries can coexist; both queues are allocated to that capacity.
The host reads just the next queue count after each level. Partition scratch
writes are read only by the block that produced them, after a block barrier.
This design requires no device recursion or dynamic parallelism.

## Rank, odd-even, and Shell

Rank sort assigns each original key i the count of keys less than it, plus the
count of equal keys at indices j < i. Index tie-breaking gives every input
key a unique destination. For `[3, -1, 3, 0]`, destinations are `[2, 0, 3, 1]`.
One thread performs N comparisons, for N² total work.

Odd-even transposition performs N global phases, alternating adjacent pairs
(0,1),(2,3),... and (1,2),(3,4),.... A pair never overlaps another pair in the
same phase; a new kernel provides the next phase's global ordering boundary.

Shell sort treats each residue modulo the gap as an independent insertion
sequence. One thread owns one residue, and kernels halve the gap until one.
Residues are disjoint within a pass. The final gap has just one sequence and
therefore just one active sorting thread.

## Classic single-thread kernels

The nine classic sorts run their complete arrays on a single GPU thread.
Their actual routines are declared host/device, so CPU tests can exercise the
same loop bodies, including duplicate handling in cycle sort and boundary
conditions in heap and cocktail sort. They demonstrate algorithm mechanics;
their presence in a CUDA program does not establish a throughput advantage.

Changing them to one independent array per thread or per block would create
a batched small-array API, which is a different interface from this package's
single-array sort.
