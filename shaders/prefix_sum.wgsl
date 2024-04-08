// https://forum.unity.com/threads/parallel-prefix-sum-computeshader.518397/#post-7887517
 
const THREADS_PER_GROUP: u32 = 512;

@group(0) @binding(5) var<storage, read> input: array<u32>;
@group(0) @binding(5) var<storage, read> output: array<u32>;

var<workgroup> bucket: array<u32, THREADS_PER_GROUP>;

void Scan(id: u32, gi: u32, x: u32) {
    bucket[gi] = x;
 
    for (var t = 1; t < THREADS_PER_GROUP; t = t << 1) {
        storageBarrier();
        let temp = bucket[gi];
        
        if (gi >= t) {
            temp += bucket[gi - t];
        }
        storageBarrier();
        bucket[gi] = temp;
    }
 
    output[id] = bucket[gi];
}
 
// Perform isolated scans within each group.

[numthreads(THREADS_PER_GROUP, 1, 1)]
void ScanInGroupsInclusive(uint id : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    uint x = 0;
    if ((int)id < N)
        x = InputBufR[id];
 
    Scan(id, gi, x);
}
 
// Scan the sums of each of the groups (partial sums) from the preceding ScanInGroupsInclusive/Exclusive call.
[numthreads(THREADS_PER_GROUP, 1, 1)]
void ScanSums(uint id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    let idx = (id * THREADS_PER_GROUP - 1);
    let x = 0;
    
    if (idx >= 0 && idx < arrayLength(&input)) {
        x = InputBufR[idx];
    }
 
    Scan(id, gi, x);
}

// Add the scanned sums to the output of the first kernel call, to get the final, complete prefix sum.

[numthreads(THREADS_PER_GROUP, 1, 1)]
void AddScannedSums(uint id : SV_DispatchThreadID, uint gid : SV_GroupID)
{
    if (id < arrayLength(&input)) {
        OutputBufW[id] += InputBufR[gid];
    }
}