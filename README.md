# CUDA-Segmented-Scan
## How to implement multiblocks segmented scan?
The implementation is very similar to normal prefix scan, but there are a few extra things.
1. For each chunk, a max scan of `flags` is computed, store to `flags_scan`.
2. Use the last value of the chunk the computed max scan above as a flag value for auxiliary array `sums`, call this `sums_flags`.
3. After finishing computing segmented scan for all chunks, segmented scan `sums` using `sums_flags` as flags.
4. Go through each chunk, if the corresponding `flags_scan` is 0, add the value from `sums` at index of this chunk less 1.
