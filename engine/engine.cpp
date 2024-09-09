//
// Created by QIAQIA on 2024/9/5.
//

#include "engine.h"

#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;
namespace Sph {
inline uint64_t CalcBucket(const Pos<2>& pos) {
    auto bucketX = static_cast<uint64_t>(pos.x[0] / Engine2D::SMOOTHING_LENGTH);
    auto bucketY = static_cast<uint64_t>(pos.x[1] / Engine2D::SMOOTHING_LENGTH);
    return bucketY + Engine2D::BUCKET_NUM_Y * bucketX;
}

Engine2D::Engine2D(int particleNum) {
    srand(0);
    pos_.resize(particleNum);
    u_.resize(particleNum);
    f_.resize(particleNum);
    p_.resize(particleNum);
    rho_.resize(particleNum);
    xyzsVec_.resize(particleNum * 4);
    colorVec_.resize(particleNum * 4);
    idxBucket_.resize(BUCKET_NUM_X * BUCKET_NUM_Y);

    for (int i = 0; i < particleNum; ++i) {
        float y = -(static_cast<float>(rand()) / RAND_MAX) * (DOMAIN_Y_LIM[1] - DOMAIN_Y_LIM[0]) / 2 + DOMAIN_Y_LIM[1];
        float x = -(static_cast<float>(rand()) / RAND_MAX) * (DOMAIN_X_LIM[1] - DOMAIN_X_LIM[0]) / 2 + DOMAIN_X_LIM[1];
        pos_[i] = {x, y};
        idxBucket_[CalcBucket(pos_[i])].emplace(i);
    }
    InitCl();
}
const std::vector<float>& Engine2D::GetColor() {
    for (uint64_t i = 0; i < rho_.size(); ++i) {
        float diff = 2 - min(max(rho_[i], 0.0f), BASE_DENSITY * 2) / BASE_DENSITY;
        colorVec_[i * 4 + 0] = diff < 1 ? 1 : 2 - diff;
        colorVec_[i * 4 + 1] = diff < 1 ? diff : 1;
        colorVec_[i * 4 + 2] = 0;
        colorVec_[i * 4 + 3] = 1;
    }
    return colorVec_;
}
const std::vector<float>& Engine2D::GetXyzs() {
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
void Engine2D::Step() {
    for (int i = 0; i < RENDER_INTERVAL; ++i) {
        StepOne();
    }
    //    cout << time_ << endl;
}

inline float Distance(const Pos<2>& pos1, const Pos<2>& pos2) {
    return sqrt((pos1.x[0] - pos2.x[0]) * (pos1.x[0] - pos2.x[0]) + (pos1.x[1] - pos2.x[1]) * (pos1.x[1] - pos2.x[1]));
}
void Engine2D::FindPairPerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t occIdx = idx; occIdx < min(occupiedBucket_.size(), idx + size); ++occIdx) {
        auto curIdx = static_cast<int64_t>(occupiedBucket_[occIdx]);
        int64_t bucketX = curIdx / BUCKET_NUM_Y;
        int64_t bucketY = curIdx % BUCKET_NUM_Y;
        particlePairs_[curIdx].resize(0);
        pairDistance_[curIdx].resize(0);
        for (int64_t osY = -1; osY <= 1; ++osY) {
            auto tgtY = bucketY + osY;
            if (tgtY >= BUCKET_NUM_Y || tgtY < 0) {
                continue;
            }
            for (int64_t osX = -1; osX <= 1; ++osX) {
                auto tgtX = bucketX + osX;
                if (tgtX >= BUCKET_NUM_X || tgtX < 0) {
                    continue;
                }
                auto tgtIdx = tgtY + BUCKET_NUM_Y * tgtX;
                for (const auto& cur: idxBucket_[curIdx]) {
                    for (const auto& tgt: idxBucket_[tgtIdx]) {
                        if (cur == tgt) {
                            //                            continue;
                        }
                        auto distance = Distance(pos_[cur], pos_[tgt]);
                        if (distance < SMOOTHING_LENGTH) {
                            particlePairs_[curIdx].emplace_back(cur, tgt);
                            pairDistance_[curIdx].emplace_back(distance);
                        }
                    }
                }
            }
        }
    }
}
void Engine2D::FindPair() {
    particlePairs_.resize(BUCKET_NUM_Y * BUCKET_NUM_X);
    pairDistance_.resize(BUCKET_NUM_Y * BUCKET_NUM_X);
    occupiedBucket_.resize(0);
    for (uint64_t i = 0; i < idxBucket_.size(); ++i) {
        if (!idxBucket_[i].empty()) {
            occupiedBucket_.emplace_back(i);
        }
    }
    auto blockNum = std::thread::hardware_concurrency() / 4 * 3;
    auto blockSize = occupiedBucket_.size() / blockNum + static_cast<uint64_t>(occupiedBucket_.size() % blockNum > 0);

    vector<future<void>> retVal;
    for (uint64_t i = 0; i < occupiedBucket_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->FindPairPerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });

    // concat result
    uint64_t cnt = 0;
    for (const auto& iter: occupiedBucket_) {
        cnt += particlePairs_[iter].size();
    }
    retVal.resize(0);
    particlePairsSingle_.resize(0);
    pairDistanceSingle_.resize(0);
    particlePairsSingle_.resize(cnt, {-1, -1});
    pairDistanceSingle_.resize(cnt, -1);
    cnt = 0;
    for (uint64_t i = 0; i < occupiedBucket_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize, cnt]() {
            uint64_t localCnt = cnt;
            for (uint64_t bucketIdx = i; bucketIdx < min(this->occupiedBucket_.size(), i + blockSize); ++bucketIdx) {
                auto idx = occupiedBucket_[bucketIdx];
                for (uint64_t j = 0; j < this->particlePairs_[idx].size(); ++j) {
                    this->particlePairsSingle_[localCnt] = this->particlePairs_[idx][j];
                    this->pairDistanceSingle_[localCnt] = this->pairDistance_[idx][j];
                    localCnt++;
                }
            }
        }));
        for (uint64_t j = i; j < min(this->occupiedBucket_.size(), i + blockSize); ++j) {
            cnt += particlePairs_[occupiedBucket_[j]].size();
        }
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}

void Engine2D::StepOne() {
    FindPair();
    //    VerifyPair();
    UpdateDensity();
    UpdatePressure();
    UpdateForce();
    UpdatePosVelocity();
    time_ += DT;
}
void Engine2D::UpdatePosVelocity() {
    auto blockNum = 2;
    auto blockSize = u_.size() / blockNum + static_cast<uint64_t>(u_.size() % blockNum > 0);
    vector<future<void>> retVal;
    mutex mut;
    for (uint64_t start = 0; start < u_.size(); start += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, start, blockSize, &mut]() {
            for (uint64_t i = start; i < min(pos_.size(), start + blockSize); ++i) {
                auto origBucket = CalcBucket(pos_[i]);
                pos_[i].x[0] += u_[i].x[0] * DT;
                pos_[i].x[1] += u_[i].x[1] * DT;
                if (pos_[i].x[0] < DOMAIN_X_LIM[0]) {
                    pos_[i].x[0] = DOMAIN_X_LIM[0];
                    u_[i].x[0] *= DAMPING_COEFFICIENT;
                }
                if (pos_[i].x[1] < DOMAIN_Y_LIM[0]) {
                    pos_[i].x[1] = DOMAIN_Y_LIM[0];
                    u_[i].x[1] *= DAMPING_COEFFICIENT;
                }
                if (pos_[i].x[0] > DOMAIN_X_LIM[1]) {
                    pos_[i].x[0] = DOMAIN_X_LIM[1];
                    u_[i].x[0] *= DAMPING_COEFFICIENT;
                }
                if (pos_[i].x[1] > DOMAIN_Y_LIM[1]) {
                    pos_[i].x[1] = DOMAIN_Y_LIM[1];
                    u_[i].x[1] *= DAMPING_COEFFICIENT;
                }
                auto newBucket = CalcBucket(pos_[i]);
                if (newBucket != origBucket) {
                    lock_guard<mutex> lock(mut);
                    idxBucket_[origBucket].erase(i);
                    idxBucket_[newBucket].emplace(i);
                }
                u_[i].x[0] += (min(MAX_ACC, max(-MAX_ACC, f_[i].x[0] / rho_[i])) + G_FORCE.x[0]) * DT;
                u_[i].x[1] += (min(MAX_ACC, max(-MAX_ACC, f_[i].x[1] / rho_[i])) + G_FORCE.x[1]) * DT;
            }
        }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void Engine2D::UpdateDensity() {
    auto blockNum = std::thread::hardware_concurrency() / 4 * 3;
    auto blockSize =
            particlePairsSingle_.size() / blockNum + static_cast<uint64_t>(particlePairsSingle_.size() % blockNum > 0);
    vector<future<void>> retVal;

    rho_.resize(0);
    rho_.resize(pos_.size(), 0);
    for (uint64_t i = 0; i < particlePairsSingle_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdateDensityPerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void Engine2D::UpdateDensityPerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, particlePairsSingle_.size()); ++i) {
        const auto& [src, tgt] = particlePairsSingle_[i];
        auto dist = pairDistanceSingle_[i];
        auto squareDiff = (SMOOTHING_LENGTH * SMOOTHING_LENGTH - dist * dist);
        rho_[src] += NORMALIZATION_DENSITY * squareDiff * squareDiff * squareDiff;
    }
}
void Engine2D::UpdatePressure() {
    auto blockNum = std::thread::hardware_concurrency() / 4 * 3;
    auto blockSize =
            particlePairsSingle_.size() / blockNum + static_cast<uint64_t>(particlePairsSingle_.size() % blockNum > 0);
    vector<future<void>> retVal;

    for (uint64_t i = 0; i < p_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdatePressurePerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void Engine2D::UpdatePressurePerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, p_.size()); ++i) {
        p_[i] = ISOTROPIC_EXPONENT * (rho_[i] - BASE_DENSITY);
    }
}

void Engine2D::UpdateForce() {
    auto blockNum = std::thread::hardware_concurrency() / 4 * 3;
    auto blockSize =
            particlePairsSingle_.size() / blockNum + static_cast<uint64_t>(particlePairsSingle_.size() % blockNum > 0);
    vector<future<void>> retVal;

    f_.resize(0);
    f_.resize(pos_.size(), {0, 0});
    for (uint64_t i = 0; i < particlePairsSingle_.size(); i += blockSize) {
        retVal.emplace_back(pool_.enqueue([this, i, blockSize] { this->UpdateForcePerBlock(i, blockSize); }));
    }
    for_each(retVal.begin(), retVal.end(), [](future<void>& iter) { iter.wait(); });
}
void Engine2D::UpdateForcePerBlock(uint64_t idx, uint64_t size) {
    for (uint64_t i = idx; i < min(idx + size, particlePairsSingle_.size()); ++i) {
        const auto& [src, tgt] = particlePairsSingle_[i];
        if (src == tgt) {
            continue;
        }
        auto distance = pairDistanceSingle_[i];
        auto force = NORMALIZATION_PRESSURE_FORCE * (p_[tgt] + p_[src]) / (2 * rho_[tgt]) *
                     (SMOOTHING_LENGTH - distance) * (SMOOTHING_LENGTH - distance);
        f_[src].x[0] += force * (pos_[tgt].x[0] - pos_[src].x[0]) / max(distance, 0.001f);
        f_[src].x[1] += force * (pos_[tgt].x[1] - pos_[src].x[1]) / max(distance, 0.001f);

        auto viscosity = NORMALIZATION_VISCOUS_FORCE * 1 / (2 * rho_[tgt]) * (SMOOTHING_LENGTH - distance);
        f_[src].x[0] += viscosity * (u_[tgt].x[0] - u_[src].x[0]);
        f_[src].x[1] += viscosity * (u_[tgt].x[1] - u_[src].x[1]);

        f_[src].x[0] = f_[src].x[0];
        f_[src].x[1] = f_[src].x[1];
    }
}
void Engine2D::VerifyPair() {
    uint64_t cnt = 0;
    for (int i = 0; i < pos_.size(); ++i) {
        for (int j = 0; j < pos_.size(); ++j) {
            if (i == j) {
                //                continue;
            }
            cnt += (Distance(pos_[i], pos_[j]) < SMOOTHING_LENGTH);
        }
    }
    if (cnt != particlePairsSingle_.size()) {
        exit(-1);
    }
    for (const auto& iter: pairDistanceSingle_) {
        if (iter < 0) {
            exit(-1);
        }
    }
}
}// namespace Sph