// #define MAX_LOCAL_SIZE 256//set via compile options
#define INT32_MAX 0x7FFFFFFF
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// needed helper methods
inline void swap(uint2 *a, uint2 *b) {
    uint2 tmp;
    tmp = *b;
    *b = *a;
    *a = tmp;
}

// dir == 1 means ascending
inline void sort(uint2 *a, uint2 *b, char dir) {
    if ((a->y > b->y) == dir) swap(a, b);
}

inline void swapLocal(__local uint2 *a, __local uint2 *b) {
    uint2 tmp;
    tmp = *b;
    *b = *a;
    *a = tmp;
}

// dir == 1 means ascending
inline void sortLocal(__local uint2 *a, __local uint2 *b, char dir) {
    if ((a->y > b->y) == dir) swapLocal(a, b);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Sort_BitonicMergesortStart(const __global uint2 *inArray, __global uint2 *outArray) {
    __local uint2 local_buffer[MAX_LOCAL_SIZE * 2];
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

__kernel void Sort_BitonicMergesortLocal(__global uint2 *data, const uint size, const uint blocksize, uint stride) {
    // This Kernel is basically the same as Sort_BitonicMergesortStart except of the "unrolled" part and the provided parameters
    __local uint2 local_buffer[2 * MAX_LOCAL_SIZE];
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

__kernel void Sort_BitonicMergesortGlobal(__global uint2 *data, const uint size, const uint blocksize,
                                          const uint stride) {
    // TO DO: Kernel implementation
    uint gid = get_global_id(0);
    uint clampedGID = gid & (size / 2 - 1);

    //calculate index and dir like above
    uint index = 2 * clampedGID - (clampedGID & (stride - 1));
    char dir = (clampedGID & (blocksize / 2)) == 0;//same as above, % calc

    //bitonic merge
    uint2 left = data[index];
    uint2 right = data[index + stride];

    sort(&left, &right, dir);

    // writeback
    data[index] = left;
    data[index + stride] = right;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline uint CalcBucketHash(const float2 pos, const uint SMOOTHING_LENGTH) {
    static const uint hashK1 = 15823;
    static const uint hashK2 = 9737333;
    static const uint hashK3 = 440817757;
    uint2 posInt = convert_uint2(pos);
    uint smooth = (uint) SMOOTHING_LENGTH;
    return (posInt.x / smooth) * hashK1 + (posInt.y / smooth) * hashK2;
}

inline uint CalcBucketHashFromBucketLoc(uint bucketX, uint bucketY) {
    static const uint hashK1 = 15823;
    static const uint hashK2 = 9737333;
    static const uint hashK3 = 440817757;
    return bucketX * hashK1 + bucketY * hashK2;
}

inline float Distance(const float2 pos1, float2 pos2) {
    return sqrt((pos1.x - pos2.x) * (pos1.x - pos2.x) + (pos1.y - pos2.y) * (pos1.y - pos2.y));
}

__kernel void UpdateHashKernel(__global float2 *pos, __global uint2 *bucket, const uint size,
                               const float SMOOTHING_LENGTH) {
    int idx = get_global_id(0);
    uint key = CalcBucketHash(pos[idx], SMOOTHING_LENGTH);
    uint hash = key % size;
    if (idx < size) {
        bucket[idx].x = idx;
        bucket[idx].y = hash;
    } else {
        bucket[idx].x = INT32_MAX;
        bucket[idx].y = INT32_MAX;
    }
}

__kernel void UpdateStartIdxKernel(__global const uint2 *bucket, __global uint *bucketKeyStartIdxMap) {
    int idx = get_global_id(0);
    if (idx == 0 || bucket[idx].y != bucket[idx - 1].y) {
        bucketKeyStartIdxMap[bucket[idx].y] = idx;
    }
}
__kernel void UpdateDensityPressureKernel(__global const uint2 *bucket, __global const uint *bucketKeyStartIdxMap,
                                          __global const float2 *pos, __global float *rho, __global float *p,
                                          const uint size, const float SMOOTHING_LENGTH,
                                          const float NORMALIZATION_DENSITY, const float ISOTROPIC_EXPONENT,
                                          const float BASE_DENSITY) {
    int idx = get_global_id(0);
    rho[idx] = 0;
    uint blockX = (uint) (pos[idx].x) / (uint) (SMOOTHING_LENGTH);
    uint blockY = (uint) (pos[idx].y) / (uint) (SMOOTHING_LENGTH);
    for (uint by = blockY - 1; by != blockY + 2; ++by) {
        for (uint bx = blockX - 1; bx != blockX + 2; ++bx) {
            uint tgtKey = CalcBucketHashFromBucketLoc(bx, by) % size;
            uint tgtKeyStartIdx = bucketKeyStartIdxMap[tgtKey];
            for (uint tgtBucketIdx = tgtKeyStartIdx; tgtBucketIdx < size; ++tgtBucketIdx) {
                if (bucket[tgtBucketIdx].y != tgtKey) {
                    break;
                }
                uint tgt = bucket[tgtBucketIdx].x;
                uint tgtBx = (uint) (pos[tgt].x) / (uint) (SMOOTHING_LENGTH);
                uint tgtBy = (uint) (pos[tgt].y) / (uint) (SMOOTHING_LENGTH);
                if (tgtBx != bx || tgtBy != by) {
                    continue;
                }
                float dist = Distance(pos[idx], pos[tgt]);
                if (dist >= SMOOTHING_LENGTH) {
                    continue;
                }
                float squareDiff = (SMOOTHING_LENGTH * SMOOTHING_LENGTH - dist * dist);
                rho[idx] += NORMALIZATION_DENSITY * squareDiff * squareDiff * squareDiff;
            }
        }
    }
    p[idx] = ISOTROPIC_EXPONENT * (rho[idx] - BASE_DENSITY);
}


__kernel void UpdateForceKernel(__global const uint2 *bucket, __global const uint *bucketKeyStartIdxMap,
                                __global const float2 *pos, __global const float *rho, const __global float *p,
                                __global float2 *u, __global float2 *f, const uint size, const float SMOOTHING_LENGTH,
                                const float NORMALIZATION_PRESSURE_FORCE, const float NORMALIZATION_VISCOUS_FORCE) {
    int idx = get_global_id(0);
    f[idx] = (float2) (0, 0);
    uint blockX = (uint) (pos[idx].x) / (uint) (SMOOTHING_LENGTH);
    uint blockY = (uint) (pos[idx].y) / (uint) (SMOOTHING_LENGTH);
    for (uint by = blockY - 1; by != blockY + 2; ++by) {
        for (uint bx = blockX - 1; bx != blockX + 2; ++bx) {
            uint tgtKey = CalcBucketHashFromBucketLoc(bx, by) % size;
            uint tgtKeyStartIdx = bucketKeyStartIdxMap[tgtKey];
            for (uint tgtBucketIdx = tgtKeyStartIdx; tgtBucketIdx < size; ++tgtBucketIdx) {
                if (bucket[tgtBucketIdx].y != tgtKey) {
                    break;
                }
                uint tgt = bucket[tgtBucketIdx].x;
                uint tgtBx = (uint) (pos[tgt].x) / (uint) (SMOOTHING_LENGTH);
                uint tgtBy = (uint) (pos[tgt].y) / (uint) (SMOOTHING_LENGTH);
                if (tgtBx != bx || tgtBy != by || tgt == idx) {
                    continue;
                }
                float dist = Distance(pos[idx], pos[tgt]);
                if (dist >= SMOOTHING_LENGTH) {
                    continue;
                }
                float force = NORMALIZATION_PRESSURE_FORCE * (p[tgt] + p[idx]) / (2 * rho[tgt]) *
                              (SMOOTHING_LENGTH - dist) * (SMOOTHING_LENGTH - dist);
                f[idx].x += force * (pos[tgt].x - pos[idx].x) / max(dist, 0.001f);
                f[idx].y += force * (pos[tgt].y - pos[idx].y) / max(dist, 0.001f);

                float viscosity = NORMALIZATION_VISCOUS_FORCE * 1 / (2 * rho[tgt]) * (SMOOTHING_LENGTH - dist);
                f[idx].x += viscosity * (u[tgt].x - u[idx].x);
                f[idx].y += viscosity * (u[tgt].y - u[idx].y);

                f[idx].x = f[idx].x;
                f[idx].y = f[idx].y;
            }
        }
    }
}

__kernel void UpdatePosVelocityKernel(__global const float *rho, __global float2 *f, __global float2 *pos,
                                      __global float2 *u, const uint size, const float2 G_FORCE,
                                      const float2 DOMAIN_X_LIM, const float2 DOMAIN_Y_LIM, const float DT,
                                      const float DAMPING_COEFFICIENT, const float MAX_ACC) {
    int idx = get_global_id(0);
    pos[idx].x += u[idx].x * DT;
    pos[idx].y += u[idx].y * DT;
    if (pos[idx].x < DOMAIN_X_LIM.x) {
        pos[idx].x = DOMAIN_X_LIM.x;
        u[idx].x *= DAMPING_COEFFICIENT;
    }
    if (pos[idx].y < DOMAIN_Y_LIM.x) {
        pos[idx].y = DOMAIN_Y_LIM.x;
        u[idx].y *= DAMPING_COEFFICIENT;
    }
    if (pos[idx].x > DOMAIN_X_LIM.y) {
        pos[idx].x = DOMAIN_X_LIM.y;
        u[idx].x *= DAMPING_COEFFICIENT;
    }
    if (pos[idx].y > DOMAIN_Y_LIM.y) {
        pos[idx].y = DOMAIN_Y_LIM.y;
        u[idx].y *= DAMPING_COEFFICIENT;
    }
    u[idx].x += (min(MAX_ACC, max(-MAX_ACC, f[idx].x / rho[idx])) + G_FORCE.x) * DT;
    u[idx].y += (min(MAX_ACC, max(-MAX_ACC, f[idx].y / rho[idx])) + G_FORCE.y) * DT;
}

__kernel void GetColorKernel(__global const float *rho, __global float *colorVec, const float BASE_DENSITY) {

    int i = get_global_id(0);
    float diff = 2 - min(max(rho[i], 0.0f), BASE_DENSITY * 2) / BASE_DENSITY;
    colorVec[i * 4 + 0] = diff < 1 ? 1 : 2 - diff;
    colorVec[i * 4 + 1] = diff < 1 ? diff : 1;
    colorVec[i * 4 + 2] = 0;
    colorVec[i * 4 + 3] = 1;
}

__kernel void GetXyzsKernel(__global const float2 *pos, __global float *xyzsVec, const float biasX, const float biasY,
                            const float scaling) {

    int i = get_global_id(0);
    xyzsVec[i * 4 + 0] = (pos[i].x + biasX) / scaling;
    xyzsVec[i * 4 + 1] = (pos[i].y + biasY) / scaling;
    xyzsVec[i * 4 + 2] = i / 10000.0f;
    xyzsVec[i * 4 + 3] = 1 / scaling;
}