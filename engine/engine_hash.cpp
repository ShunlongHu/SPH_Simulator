//
// Created by QIAQIA on 2024/9/5.
//

#include "engine_hash.h"

#include <algorithm>
#include <fstream>
#include <iostream>

#define VERIFY_SORT true

using namespace std;
namespace Sph {
inline uint32_t CalcBucketHash(const Pos<2>& pos) {
    static const uint32_t hashK1 = 15823;
    static const uint32_t hashK2 = 9737333;
    static const uint32_t hashK3 = 440817757;
    return pos.x[0] * hashK1 + pos.x[1] * hashK2;
}

inline uint32_t CalcBucketHash(uint32_t x, uint32_t y) {
    static const uint32_t hashK1 = 15823;
    static const uint32_t hashK2 = 9737333;
    static const uint32_t hashK3 = 440817757;
    return x * hashK1 + y * hashK2;
}


EngineHash2D::EngineHash2D(int particleNum) {
    srand(0);
    pos_.resize(particleNum);
    u_.resize(particleNum);
    f_.resize(particleNum);
    p_.resize(particleNum);
    rho_.resize(particleNum);
    xyzsVec_.resize(particleNum * 4);
    colorVec_.resize(particleNum * 4);

    bucketIdxIdxMap_.resize(particleNum);
    bucketKeyStartIdxMap_.resize(particleNum);
    bucket_.resize(particleNum);
    unsortedBucket_.resize(particleNum);

    for (int i = 0; i < particleNum; ++i) {
        float y = -(static_cast<float>(rand()) / RAND_MAX) * (DOMAIN_Y_LIM[1] - DOMAIN_Y_LIM[0]) / 2 + DOMAIN_Y_LIM[1];
        float x = -(static_cast<float>(rand()) / RAND_MAX) * (DOMAIN_X_LIM[1] - DOMAIN_X_LIM[0]) / 2 + DOMAIN_X_LIM[1];
        pos_[i] = {x, y};
    }
}

const std::vector<float>& EngineHash2D::GetColor() {
    for (uint64_t i = 0; i < rho_.size(); ++i) {
        float diff = 2 - min(max(rho_[i], 0.0f), BASE_DENSITY * 2) / BASE_DENSITY;
        colorVec_[i * 4 + 0] = diff < 1 ? 1 : 2 - diff;
        colorVec_[i * 4 + 1] = diff < 1 ? diff : 1;
        colorVec_[i * 4 + 2] = 0;
        colorVec_[i * 4 + 3] = 1;
    }
    return colorVec_;
}
const std::vector<float>& EngineHash2D::GetXyzs() {
    float scaling = max(DOMAIN_HEIGHT, DOMAIN_WIDTH) / 4.0f;
    float biasX = DOMAIN_WIDTH / -2.0f;
    float biasY = DOMAIN_HEIGHT / -2.0f;
    for (uint64_t i = 0; i < pos_.size(); ++i) {
        xyzsVec_[i * 4 + 0] = (pos_[i].x[0] + biasX) / scaling;
        xyzsVec_[i * 4 + 1] = (pos_[i].x[1] + biasY) / scaling;
        xyzsVec_[i * 4 + 2] = i / 10000.0f;
        xyzsVec_[i * 4 + 3] = 1 / scaling;
    }
    return xyzsVec_;
}
void EngineHash2D::Step() {
    for (int i = 0; i < RENDER_INTERVAL; ++i) {
        StepOne();
    }
    //    cout << time_ << endl;
}

inline float Distance(const Pos<2>& pos1, const Pos<2>& pos2) {
    return sqrt((pos1.x[0] - pos2.x[0]) * (pos1.x[0] - pos2.x[0]) + (pos1.x[1] - pos2.x[1]) * (pos1.x[1] - pos2.x[1]));
}

struct BitonicParam {
    uint32_t groupWidth;
    uint32_t groupHeight;
    uint32_t stepIndex;
    uint32_t numEntries;
};

inline void BitonicMergeSortKernel(vector<uint32_t>& value, vector<uint32_t>& idx, uint32_t i, const BitonicParam& p) {
    uint32_t hIndex = i & (p.groupWidth - 1);
    uint32_t indexLeft = hIndex + (p.groupHeight + 1) * (i / p.groupWidth);
    uint32_t rightStepSize = p.stepIndex == 0 ? p.groupHeight - 2 * hIndex : (p.groupHeight + 1) / 2;
    uint32_t indexRight = indexLeft + rightStepSize;

    // Exit if out of bounds (for non-power of 2 input sizes)
    if (indexRight >= p.numEntries) return;

    uint32_t valueLeft = value[indexLeft];
    uint32_t valueRight = value[indexRight];

    // Swap entries if value is descending
    if (valueLeft > valueRight) {
        auto tVal = value[indexLeft];
        value[indexLeft] = value[indexRight];
        value[indexRight] = tVal;
        auto tIdx = idx[indexLeft];
        idx[indexLeft] = idx[indexRight];
        idx[indexRight] = tIdx;
    }
}

void BitonicMergeSortBlock(vector<uint32_t>& value, vector<uint32_t>& idx, uint32_t start, uint32_t size) {
    for (uint32_t i = start; i < min<uint32_t>(value.size(), start); ++i) {
        BitoniceMergeSortKernel(value, idx, i);
    }
}

inline void BitonicMergeSort(vector<uint32_t>& value, vector<uint32_t>& idx) {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = value.size() / blockNum + static_cast<uint64_t>(value.size() % blockNum > 0);
    BitonicParam p{};
    vector<future<void>> retVal;
    p.numEntries = value.size();
    auto numStages = Log(NextPowerOfTwo(indexBuffer.count), 2);

    for (uint32_t i = 0; i < value.size(); i += blockSize) {
        BitonicMergeSortBlock(value, idx, i, blockSize);
    }


    for (uint32_t step = 2; step < value.size() * 2; step *= 2) {
        BitonicMergeSortBlock(value, idx, cacheValue, cacheIdx, step);
    }
}
inline void VerifySort(const vector<uint32_t>& value, const vector<uint32_t>& idx, const vector<uint32_t>& origValue) {
    for (uint32_t i = 1; i < value.size(); ++i) {
        if (value[i] < value[i - 1]) {
            cerr << "Sort Incorrect" << endl;
            exit(-1);
        }
    }
    for (uint32_t i = 0; i < value.size(); ++i) {
        if (value[i] != origValue[idx[i]]) {
            cerr << "Sort Incorrect" << endl;
            exit(-1);
        }
    }
}

void CalcStartIdx(const vector<uint32_t>& bucket, vector<uint32_t>& startIdx, uint32_t idx) {
    if (idx == 0 || bucket[idx] != bucket[idx - 1]) {
        startIdx[bucket[idx]] = idx;
    }
}

void EngineHash2D::UpdateBucket() {
    for (int i = 0; i < pos_.size(); ++i) {
        auto key = CalcBucketHash(pos_[i]);
        auto hash = key % pos_.size();
        bucketIdxIdxMap_[i] = i;
        bucket_[i] = hash;
        bucketKeyStartIdxMap_[i] = INT32_MAX;
        if (VERIFY_SORT) {
            unsortedBucket_[i] = hash;
        }
    }

    BitonicMergeSort(bucket_, bucketIdxIdxMap_);
    if (VERIFY_SORT) {
        VerifySort(bucket_, bucketIdxIdxMap_, unsortedBucket_);
    }

    uint32_t lastVal = INT32_MAX;
    for (int i = 0; i < bucket_.size(); ++i) {
        CalcStartIdx(bucket_, bucketKeyStartIdxMap_, i);
    }
}

void EngineHash2D::StepOne() {
    UpdateBucket();
    UpdateDensity();
    UpdatePressure();
    UpdateForce();
    UpdatePosVelocity();
    time_ += DT;
}
void EngineHash2D::UpdatePosVelocity() {
    //    auto blockNum = 2;
    //    auto blockSize = u_.size() / blockNum + static_cast<uint64_t>(u_.size() % blockNum > 0);
    //    vector<future<void>> retVal;
    //    mutex mut;
    //    for (uint64_t start = 0; start < u_.size(); start += blockSize) {
    //        retVal.emplace_back(pool_.enqueue([this, start, blockSize, &mut]() {
    //            for (uint64_t i = start; i < min(pos_.size(), start + blockSize); ++i) {
    //                auto origBucket = CalcBucketHash(pos_[i]);
    //                pos_[i].x[0] += u_[i].x[0] * DT;
    //                pos_[i].x[1] += u_[i].x[1] * DT;
    //                if (pos_[i].x[0] < DOMAIN_X_LIM[0]) {
    //                    pos_[i].x[0] = DOMAIN_X_LIM[0];
    //                    u_[i].x[0] *= DAMPING_COEFFICIENT;
    //                }
    //                if (pos_[i].x[1] < DOMAIN_Y_LIM[0]) {
    //                    pos_[i].x[1] = DOMAIN_Y_LIM[0];
    //                    u_[i].x[1] *= DAMPING_COEFFICIENT;
    //                }
    //                if (pos_[i].x[0] > DOMAIN_X_LIM[1]) {
    //                    pos_[i].x[0] = DOMAIN_X_LIM[1];
    //                    u_[i].x[0] *= DAMPING_COEFFICIENT;
    //                }
    //                if (pos_[i].x[1] > DOMAIN_Y_LIM[1]) {
    //                    pos_[i].x[1] = DOMAIN_Y_LIM[1];
    //                    u_[i].x[1] *= DAMPING_COEFFICIENT;
    //                }
    //                auto newBucket = CalcBucketHash(pos_[i]);
    //                if (newBucket != origBucket) {
    //                    lock_guard<mutex> lock(mut);
    //                    idxBucket_[origBucket].erase(i);
    //                    idxBucket_[newBucket].emplace(i);
    //                }
    //                u_[i].x[0] += (min(MAX_ACC, max(-MAX_ACC, f_[i].x[0] / rho_[i])) + G_FORCE.x[0]) * DT;
    //                u_[i].x[1] += (min(MAX_ACC, max(-MAX_ACC, f_[i].x[1] / rho_[i])) + G_FORCE.x[1]) * DT;
    //            }
    //        }));
    //    }
    //    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void EngineHash2D::UpdateDensity() {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = pos_.size() / blockNum + static_cast<uint64_t>(pos_.size() % blockNum > 0);
    vector<future<void>> retVal;

    rho_.resize(0);
    rho_.resize(pos_.size(), 0);
    for (uint64_t i = 0; i < pos_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdateDensityPerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void EngineHash2D::UpdateDensityPerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, pos_.size()); ++i) {
        UpdateDensityKernel(idx);
    }
}
void EngineHash2D::UpdateDensityKernel(uint64_t idx) {
    const auto& pos = pos_[idx];
    auto blockX = static_cast<int32_t>(pos.x[0]) / static_cast<int32_t>(SMOOTHING_LENGTH);
    auto blockY = static_cast<int32_t>(pos.x[1]) / static_cast<int32_t>(SMOOTHING_LENGTH);
    for (uint32_t by = blockY - 1; by <= blockX + 1; ++by) {
        for (uint32_t bx = blockX - 1; bx <= blockX + 1; ++bx) {
            auto tgtKey = CalcBucketHash(bx, by) % pos_.size();
            auto tgtKeyStartIdx = bucketKeyStartIdxMap_[tgtKey];
            for (uint32_t tgtBucketIdx = tgtKeyStartIdx; tgtBucketIdx < bucketKeyStartIdxMap_.size(); ++tgtBucketIdx) {
                if (bucket_[tgtBucketIdx] != tgtKey) {
                    break;
                }
            }
        }
    }
}

//void EngineHash2D::UpdateDensityPerBlock(uint64_t idx, uint64_t size) {
//    for (uint64_t i = idx; i < min(idx + size, particlePairsSingle_.size()); ++i) {
//        const auto& [src, tgt] = particlePairsSingle_[i];
//        auto dist = pairDistanceSingle_[i];
//        auto squareDiff = (SMOOTHING_LENGTH * SMOOTHING_LENGTH - dist * dist);
//        rho_[src] += NORMALIZATION_DENSITY * squareDiff * squareDiff * squareDiff;
//    }
//}
void EngineHash2D::UpdatePressure() {
    //    auto blockNum = std::thread::hardware_concurrency() / 4 * 3;
    //    auto blockSize =
    //            particlePairsSingle_.size() / blockNum + static_cast<uint64_t>(particlePairsSingle_.size() % blockNum > 0);
    //    vector<future<void>> retVal;
    //
    //    for (uint64_t i = 0; i < p_.size(); i += blockSize) {
    //        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdatePressurePerBlock(i, blockSize); }));
    //    }
    //    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
//void EngineHash2D::UpdatePressurePerBlock(uint64_t idx, uint64_t size) {
//    for (uint64_t i = idx; i < min(idx + size, p_.size()); ++i) {
//        p_[i] = ISOTROPIC_EXPONENT * (rho_[i] - BASE_DENSITY);
//    }
//}

void EngineHash2D::UpdateForce() {
    //    auto blockNum = std::thread::hardware_concurrency() / 4 * 3;
    //    auto blockSize =
    //            particlePairsSingle_.size() / blockNum + static_cast<uint64_t>(particlePairsSingle_.size() % blockNum > 0);
    //    vector<future<void>> retVal;
    //
    //    f_.resize(0);
    //    f_.resize(pos_.size(), {0, 0});
    //    for (uint64_t i = 0; i < particlePairsSingle_.size(); i += blockSize) {
    //        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdateForcePerBlock(i, blockSize); }));
    //    }
    //    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
//void EngineHash2D::UpdateForcePerBlock(uint64_t idx, uint64_t size) {
//    for (uint64_t i = idx; i < min(idx + size, particlePairsSingle_.size()); ++i) {
//        const auto& [src, tgt] = particlePairsSingle_[i];
//        if (src == tgt) {
//            continue;
//        }
//        auto distance = pairDistanceSingle_[i];
//        auto force = NORMALIZATION_PRESSURE_FORCE * (p_[tgt] + p_[src]) / (2 * rho_[tgt]) *
//                     (SMOOTHING_LENGTH - distance) * (SMOOTHING_LENGTH - distance);
//        f_[src].x[0] += force * (pos_[tgt].x[0] - pos_[src].x[0]) / max(distance, 0.001f);
//        f_[src].x[1] += force * (pos_[tgt].x[1] - pos_[src].x[1]) / max(distance, 0.001f);
//
//        auto viscosity = NORMALIZATION_VISCOUS_FORCE * 1 / (2 * rho_[tgt]) * (SMOOTHING_LENGTH - distance);
//        f_[src].x[0] += viscosity * (u_[tgt].x[0] - u_[src].x[0]);
//        f_[src].x[1] += viscosity * (u_[tgt].x[1] - u_[src].x[1]);
//
//        f_[src].x[0] = f_[src].x[0];
//        f_[src].x[1] = f_[src].x[1];
//    }
//}
}// namespace Sph