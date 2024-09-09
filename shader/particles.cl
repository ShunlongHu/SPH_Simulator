#define MAX_LOCAL_SIZE 256//set via compile options

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// needed helper methods
inline void swap(uint *a, uint *b) {
    uint tmp;
    tmp = *b;
    *b = *a;
    *a = tmp;
}

// dir == 1 means ascending
inline void sort(uint *a, uint *b, char dir) {
    if ((*a > *b) == dir) swap(a, b);
}

inline void swapLocal(__local uint *a, __local uint *b) {
    uint tmp;
    tmp = *b;
    *b = *a;
    *a = tmp;
}

// dir == 1 means ascending
inline void sortLocal(__local uint *a, __local uint *b, char dir) {
    if ((*a > *b) == dir) swapLocal(a, b);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Sort_BitonicMergesortStart(const __global uint *inArray, __global uint *outArray) {
    __local uint local_buffer[MAX_LOCAL_SIZE * 2];
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);

    uint index = get_group_id(0) * (MAX_LOCAL_SIZE * 2) + lid;
    //load into local mem
    local_buffer[lid] = inArray[index];
    local_buffer[lid + MAX_LOCAL_SIZE] = inArray[index + MAX_LOCAL_SIZE];

    uint clampedGID = gid & (MAX_LOCAL_SIZE - 1);

    // bitonic merge
    for (uint blocksize = 2; blocksize < MAX_LOCAL_SIZE * 2; blocksize <<= 1) {
        char dir = (clampedGID & (blocksize / 2)) == 0;// sort every other block in the other direction (faster % calc)
#pragma unroll
        for (uint stride = blocksize >> 1; stride > 0; stride >>= 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            uint idx =
                    2 * lid - (lid & (stride - 1));//take every other input BUT starting neighbouring within one block
            sortLocal(&local_buffer[idx], &local_buffer[idx + stride], dir);
        }
    }

    // bitonic merge for biggest group is special (unrolling this so we dont need ifs in the part above)
    char dir = (clampedGID & 0);//even or odd? sort accordingly
#pragma unroll
    for (uint stride = MAX_LOCAL_SIZE; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint idx = 2 * lid - (lid & (stride - 1));
        sortLocal(&local_buffer[idx], &local_buffer[idx + stride], dir);
    }

    // sync and write back
    barrier(CLK_LOCAL_MEM_FENCE);
    outArray[index] = local_buffer[lid];
    outArray[index + MAX_LOCAL_SIZE] = local_buffer[lid + MAX_LOCAL_SIZE];
}

__kernel void Sort_BitonicMergesortLocal(__global uint *data, const uint size, const uint blocksize, uint stride) {
    // This Kernel is basically the same as Sort_BitonicMergesortStart except of the "unrolled" part and the provided parameters
    __local uint local_buffer[2 * MAX_LOCAL_SIZE];
    uint gid = get_global_id(0);
    uint groupId = get_group_id(0);
    uint lid = get_local_id(0);
    uint clampedGID = gid & (size / 2 - 1);

    uint index = groupId * (MAX_LOCAL_SIZE * 2) + lid;
    //load into local mem
    local_buffer[lid] = data[index];
    local_buffer[lid + MAX_LOCAL_SIZE] = data[index + MAX_LOCAL_SIZE];

    // bitonic merge
    char dir = (clampedGID & (blocksize / 2)) == 0;//same as above, % calc
#pragma unroll
    for (; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint idx = 2 * lid - (lid & (stride - 1));
        sortLocal(&local_buffer[idx], &local_buffer[idx + stride], dir);
    }

    // sync and write back
    barrier(CLK_LOCAL_MEM_FENCE);
    data[index] = local_buffer[lid];
    data[index + MAX_LOCAL_SIZE] = local_buffer[lid + MAX_LOCAL_SIZE];
}

__kernel void Sort_BitonicMergesortGlobal(__global uint *data, const uint size, const uint blocksize,
                                          const uint stride) {
    // TO DO: Kernel implementation
    uint gid = get_global_id(0);
    uint clampedGID = gid & (size / 2 - 1);

    //calculate index and dir like above
    uint index = 2 * clampedGID - (clampedGID & (stride - 1));
    char dir = (clampedGID & (blocksize / 2)) == 0;//same as above, % calc

    //bitonic merge
    uint left = data[index];
    uint right = data[index + stride];

    sort(&left, &right, dir);

    // writeback
    data[index] = left;
    data[index + stride] = right;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////