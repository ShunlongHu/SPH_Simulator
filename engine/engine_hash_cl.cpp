//
// Created by QIAQIA on 2024/9/5.
//

#include "engine_hash_cl.h"

#include <algorithm>
#include <fstream>
#include <iostream>

//#define VERIFY_SORT
using namespace std;
namespace Sph {
inline uint32_t CalcBucketHash(const Pos<2>& pos) {
    static const uint32_t hashK1 = 15823;
    static const uint32_t hashK2 = 9737333;
    static const uint32_t hashK3 = 440817757;
    return (static_cast<uint32_t>(pos.x[0]) / static_cast<uint32_t>(EngineHashCL2D::SMOOTHING_LENGTH)) * hashK1 +
           (static_cast<uint32_t>(pos.x[1]) / static_cast<uint32_t>(EngineHashCL2D::SMOOTHING_LENGTH)) * hashK2;
}

inline uint32_t CalcBucketHash(uint32_t bucketX, uint32_t bucketY) {
    static const uint32_t hashK1 = 15823;
    static const uint32_t hashK2 = 9737333;
    static const uint32_t hashK3 = 440817757;
    return bucketX * hashK1 + bucketY * hashK2;
}


EngineHashCL2D::EngineHashCL2D(int particleNum) {
    srand(0);
    pos_.resize(particleNum);
    u_.resize(particleNum);
    f_.resize(particleNum);
    p_.resize(particleNum);
    rho_.resize(particleNum);
    xyzsVec_.resize(particleNum * 4);
    colorVec_.resize(particleNum * 4);

    bucketKeyStartIdxMap_.resize(particleNum);
    bucket_.resize(particleNum);
    unsortedBucket_.resize(particleNum);

    for (int i = 0; i < particleNum; ++i) {
        float y = -(static_cast<float>(rand()) / RAND_MAX) * (DOMAIN_Y_LIM[1] - DOMAIN_Y_LIM[0]) / 2 + DOMAIN_Y_LIM[1];
        float x = -(static_cast<float>(rand()) / RAND_MAX) * (DOMAIN_X_LIM[1] - DOMAIN_X_LIM[0]) / 2 + DOMAIN_X_LIM[1];
        pos_[i] = {x, y};
    }
}

const std::vector<float>& EngineHashCL2D::GetColor() {
    for (uint64_t i = 0; i < rho_.size(); ++i) {
        float diff = 2 - min(max(rho_[i], 0.0f), BASE_DENSITY * 2) / BASE_DENSITY;
        colorVec_[i * 4 + 0] = diff < 1 ? 1 : 2 - diff;
        colorVec_[i * 4 + 1] = diff < 1 ? diff : 1;
        colorVec_[i * 4 + 2] = 0;
        colorVec_[i * 4 + 3] = 1;
    }
    return colorVec_;
}
const std::vector<float>& EngineHashCL2D::GetXyzs() {
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
void EngineHashCL2D::Step() {
    for (int i = 0; i < RENDER_INTERVAL; ++i) {
        StepOne();
    }
    //    cout << time_ << endl;
}

inline float Distance(const Pos<2>& pos1, const Pos<2>& pos2) {
    return sqrt((pos1.x[0] - pos2.x[0]) * (pos1.x[0] - pos2.x[0]) + (pos1.x[1] - pos2.x[1]) * (pos1.x[1] - pos2.x[1]));
}

inline void BitonicMergeSortKernel(vector<std::pair<uint32_t, uint32_t>>& value, uint32_t i, uint32_t compareBlockSize,
                                   uint32_t compareBlockSizePow, uint32_t dirChangePerIdxPow) {
    auto blockIdx = i >> (compareBlockSizePow - 1);
    bool dir = (i >> dirChangePerIdxPow) & 1;
    auto changeIdx = i & ((compareBlockSize >> 1) - 1);
    auto lessIdx = dir == 0 ? (blockIdx << compareBlockSizePow) + changeIdx
                            : (((blockIdx + 1) << compareBlockSizePow) - 1) - changeIdx;
    auto greaterIdx = dir == 0 ? lessIdx + (compareBlockSize >> 1) : lessIdx - (compareBlockSize >> 1);
    if (value[lessIdx].second > value[greaterIdx].second) {
        auto tVal = value[lessIdx];
        value[lessIdx] = value[greaterIdx];
        value[greaterIdx] = tVal;
    }
    //    if (dir == 0) {
    //        cout << lessIdx << " -> " << greaterIdx << " ; ";
    //    } else {
    //        cout << greaterIdx << " <- " << lessIdx << " ; ";
    //    }
}

void BitonicMergeSortBlock(vector<std::pair<uint32_t, uint32_t>>& value, uint32_t start, uint32_t size,
                           uint32_t compareBlockSize, uint32_t compareBlockSizePow, uint32_t dirChangePerIdxPow) {
    auto stop = min<uint32_t>(value.size() / 2, start + size);
    for (uint32_t i = start; i < stop; ++i) {
        BitonicMergeSortKernel(value, i, compareBlockSize, compareBlockSizePow, dirChangePerIdxPow);
    }
}

inline void BitonicMergeSort(vector<std::pair<uint32_t, uint32_t>>& value, ThreadPool& pool) {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = value.size() / blockNum + static_cast<uint64_t>(value.size() % blockNum > 0);
    vector<future<void>> retVal;
    retVal.reserve(blockNum);
    uint32_t nextPowOfTwo;
    uint32_t stage = 1;
    for (nextPowOfTwo = 2; nextPowOfTwo < value.size(); nextPowOfTwo *= 2) {
        stage++;
    }
    auto origSize = value.size();
    value.resize(nextPowOfTwo, {UINT32_MAX, UINT32_MAX});
    uint32_t dirChangePerIdxPow = 0;
    for (uint32_t i = 1; i < stage + 1; ++i) {
        uint32_t compareBlockSize = 1 << i;
        uint32_t compareBlockSizePow = i;
        for (uint32_t j = 0; j < i; ++j) {
            retVal.resize(0);
            for (uint32_t start = 0; start < value.size() / 2; start += blockSize) {
                retVal.emplace_back(pool.enqueue(BitonicMergeSortBlock, std::ref(value), start, blockSize,
                                                 compareBlockSize, compareBlockSizePow, dirChangePerIdxPow));
            }
            //            cout << endl;
            compareBlockSize /= 2;
            compareBlockSizePow -= 1;
            for (auto& iter: retVal) {
                iter.wait();
            }
        }
        dirChangePerIdxPow++;
    }


    value.resize(origSize);
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

inline void CalcStartIdx(const vector<std::pair<uint32_t, uint32_t>>& bucket, vector<uint32_t>& startIdx,
                         uint32_t idx) {
    if (idx == 0 || bucket[idx].second != bucket[idx - 1].second) {
        startIdx[bucket[idx].second] = idx;
    }
}

void EngineHashCL2D::UpdateBucket() {
    UpdateHash();
    BitonicMergeSort(bucket_, pool_);
#ifdef VERIFY_SORT
    VerifySort(bucket_, bucketIdxIdxMap_, unsortedBucket_);
#endif
    UpdateStartIdx();
}

void EngineHashCL2D::UpdateHash() {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = pos_.size() / blockNum + static_cast<uint64_t>(pos_.size() % blockNum > 0);
    vector<future<void>> retVal;

    for (uint64_t i = 0; i < pos_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdateHashPerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void EngineHashCL2D::UpdateHashPerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, pos_.size()); ++i) {
        UpdateHashKernel(i);
    }
}
void EngineHashCL2D::UpdateHashKernel(uint64_t idx) {
    auto key = CalcBucketHash(pos_[idx]);
    auto hash = key % pos_.size();
    bucket_[idx].first = idx;
    bucket_[idx].second = hash;
    bucketKeyStartIdxMap_[idx] = INT32_MAX;
#ifdef VERIFY_SORT
    unsortedBucket_[idx] = hash;
#endif
}

void EngineHashCL2D::UpdateStartIdx() {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = pos_.size() / blockNum + static_cast<uint64_t>(pos_.size() % blockNum > 0);
    vector<future<void>> retVal;

    for (uint64_t i = 0; i < pos_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdateStartIdxPerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void EngineHashCL2D::UpdateStartIdxPerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, pos_.size()); ++i) {
        UpdateStartIdxKernel(i);
    }
}
void EngineHashCL2D::UpdateStartIdxKernel(uint64_t idx) { CalcStartIdx(bucket_, bucketKeyStartIdxMap_, idx); }

void EngineHashCL2D::StepOne() {
    UpdateBucket();
    UpdateDensity();
    UpdatePressure();
    UpdateForce();
    UpdatePosVelocity();
    time_ += DT;
}
void EngineHashCL2D::UpdatePosVelocity() {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = pos_.size() / blockNum + static_cast<uint64_t>(pos_.size() % blockNum > 0);
    vector<future<void>> retVal;

    for (uint64_t i = 0; i < pos_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdatePosVelocityPerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void EngineHashCL2D::UpdatePosVelocityPerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, pos_.size()); ++i) {
        UpdatePosVelocityKernel(i);
    }
}
void EngineHashCL2D::UpdatePosVelocityKernel(uint64_t idx) {
    pos_[idx].x[0] += u_[idx].x[0] * DT;
    pos_[idx].x[1] += u_[idx].x[1] * DT;
    if (pos_[idx].x[0] < DOMAIN_X_LIM[0]) {
        pos_[idx].x[0] = DOMAIN_X_LIM[0];
        u_[idx].x[0] *= DAMPING_COEFFICIENT;
    }
    if (pos_[idx].x[1] < DOMAIN_Y_LIM[0]) {
        pos_[idx].x[1] = DOMAIN_Y_LIM[0];
        u_[idx].x[1] *= DAMPING_COEFFICIENT;
    }
    if (pos_[idx].x[0] > DOMAIN_X_LIM[1]) {
        pos_[idx].x[0] = DOMAIN_X_LIM[1];
        u_[idx].x[0] *= DAMPING_COEFFICIENT;
    }
    if (pos_[idx].x[1] > DOMAIN_Y_LIM[1]) {
        pos_[idx].x[1] = DOMAIN_Y_LIM[1];
        u_[idx].x[1] *= DAMPING_COEFFICIENT;
    }
    u_[idx].x[0] += (min(MAX_ACC, max(-MAX_ACC, f_[idx].x[0] / rho_[idx])) + G_FORCE.x[0]) * DT;
    u_[idx].x[1] += (min(MAX_ACC, max(-MAX_ACC, f_[idx].x[1] / rho_[idx])) + G_FORCE.x[1]) * DT;
}
void EngineHashCL2D::UpdateDensity() {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = pos_.size() / blockNum + static_cast<uint64_t>(pos_.size() % blockNum > 0);
    vector<future<void>> retVal;

    for (uint64_t i = 0; i < pos_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdateDensityPerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void EngineHashCL2D::UpdateDensityPerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, pos_.size()); ++i) {
        UpdateDensityKernel(i);
    }
}
void EngineHashCL2D::UpdateDensityKernel(uint64_t idx) {
    rho_[idx] = 0;
    const auto& pos = pos_[idx];
    auto blockX = static_cast<uint32_t>(pos.x[0]) / static_cast<uint32_t>(SMOOTHING_LENGTH);
    auto blockY = static_cast<uint32_t>(pos.x[1]) / static_cast<uint32_t>(SMOOTHING_LENGTH);
    for (uint32_t by = blockY - 1; by <= blockY + 1; ++by) {
        for (uint32_t bx = blockX - 1; bx <= blockX + 1; ++bx) {
            auto tgtKey = CalcBucketHash(bx, by) % pos_.size();
            auto tgtKeyStartIdx = bucketKeyStartIdxMap_[tgtKey];
            for (uint32_t tgtBucketIdx = tgtKeyStartIdx; tgtBucketIdx < bucketKeyStartIdxMap_.size(); ++tgtBucketIdx) {
                if (bucket_[tgtBucketIdx].second != tgtKey) {
                    break;
                }
                auto tgt = bucket_[tgtBucketIdx].first;
                auto tgtBx = static_cast<uint32_t>(pos_[tgt].x[0]) / static_cast<uint32_t>(SMOOTHING_LENGTH);
                auto tgtBy = static_cast<uint32_t>(pos_[tgt].x[1]) / static_cast<uint32_t>(SMOOTHING_LENGTH);
                if (tgtBx != bx || tgtBy != by) {
                    continue;
                }
                auto dist = Distance(pos_[idx], pos_[tgt]);
                if (dist >= SMOOTHING_LENGTH) {
                    continue;
                }
                auto squareDiff = (SMOOTHING_LENGTH * SMOOTHING_LENGTH - dist * dist);
                rho_[idx] += NORMALIZATION_DENSITY * squareDiff * squareDiff * squareDiff;
            }
        }
    }
}

void EngineHashCL2D::UpdatePressure() {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = pos_.size() / blockNum + static_cast<uint64_t>(pos_.size() % blockNum > 0);
    vector<future<void>> retVal;

    for (uint64_t i = 0; i < pos_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdatePressurePerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void EngineHashCL2D::UpdatePressurePerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, pos_.size()); ++i) {
        UpdatePressureKernel(i);
    }
}
void EngineHashCL2D::UpdatePressureKernel(uint64_t idx) { p_[idx] = ISOTROPIC_EXPONENT * (rho_[idx] - BASE_DENSITY); }

void EngineHashCL2D::UpdateForce() {
    auto blockNum = std::thread::hardware_concurrency();
    auto blockSize = pos_.size() / blockNum + static_cast<uint64_t>(pos_.size() % blockNum > 0);
    vector<future<void>> retVal;

    for (uint64_t i = 0; i < pos_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdateForcePerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void EngineHashCL2D::UpdateForcePerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, pos_.size()); ++i) {
        UpdateForceKernel(i);
    }
}
void EngineHashCL2D::UpdateForceKernel(uint64_t idx) {
    f_[idx] = {0, 0};
    const auto& pos = pos_[idx];
    auto blockX = static_cast<uint32_t>(pos.x[0]) / static_cast<uint32_t>(SMOOTHING_LENGTH);
    auto blockY = static_cast<uint32_t>(pos.x[1]) / static_cast<uint32_t>(SMOOTHING_LENGTH);
    for (uint32_t by = blockY - 1; by <= blockY + 1; ++by) {
        for (uint32_t bx = blockX - 1; bx <= blockX + 1; ++bx) {
            auto tgtKey = CalcBucketHash(bx, by) % pos_.size();
            auto tgtKeyStartIdx = bucketKeyStartIdxMap_[tgtKey];
            for (uint32_t tgtBucketIdx = tgtKeyStartIdx; tgtBucketIdx < bucketKeyStartIdxMap_.size(); ++tgtBucketIdx) {
                if (bucket_[tgtBucketIdx].second != tgtKey) {
                    break;
                }
                auto tgt = bucket_[tgtBucketIdx].first;
                auto tgtBx = static_cast<uint32_t>(pos_[tgt].x[0]) / static_cast<uint32_t>(SMOOTHING_LENGTH);
                auto tgtBy = static_cast<uint32_t>(pos_[tgt].x[1]) / static_cast<uint32_t>(SMOOTHING_LENGTH);
                if (tgtBx != bx || tgtBy != by) {
                    continue;
                }
                if (tgt == idx) {
                    continue;
                }
                auto dist = Distance(pos_[idx], pos_[tgt]);
                if (dist >= SMOOTHING_LENGTH) {
                    continue;
                }
                auto force = NORMALIZATION_PRESSURE_FORCE * (p_[tgt] + p_[idx]) / (2 * rho_[tgt]) *
                             (SMOOTHING_LENGTH - dist) * (SMOOTHING_LENGTH - dist);
                f_[idx].x[0] += force * (pos_[tgt].x[0] - pos_[idx].x[0]) / max(dist, 0.001f);
                f_[idx].x[1] += force * (pos_[tgt].x[1] - pos_[idx].x[1]) / max(dist, 0.001f);

                auto viscosity = NORMALIZATION_VISCOUS_FORCE * 1 / (2 * rho_[tgt]) * (SMOOTHING_LENGTH - dist);
                f_[idx].x[0] += viscosity * (u_[tgt].x[0] - u_[idx].x[0]);
                f_[idx].x[1] += viscosity * (u_[tgt].x[1] - u_[idx].x[1]);

                f_[idx].x[0] = f_[idx].x[0];
                f_[idx].x[1] = f_[idx].x[1];
            }
        }
    }
}
}// namespace Sph