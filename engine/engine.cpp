//
// Created by QIAQIA on 2024/9/5.
//

#include "engine.h"

#include <algorithm>
#include <iostream>
using namespace std;
namespace Sph {
inline uint64_t CalcBucket(const Pos<2>& pos) {
    auto bucketX = static_cast<uint64_t>(pos.x[0] / Engine2D::SMOOTHING_LENGTH);
    auto bucketY = static_cast<uint64_t>(pos.x[1] / Engine2D::SMOOTHING_LENGTH);
    return bucketY * Engine2D::BUCKET_NUM_X + bucketX;
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
        xyzsVec_[i * 4 + 3] = SMOOTHING_LENGTH / scaling;
    }
    return xyzsVec_;
}
void Engine2D::Step() {
    for (int i = 0; i < RENDER_INTERVAL; ++i) {
        StepOne();
    }
    cout << time_ << endl;
}

inline float Distance(const Pos<2>& pos1, const Pos<2>& pos2) {
    return sqrt((pos1.x[0] - pos2.x[0]) * (pos1.x[0] - pos2.x[0]) + (pos1.x[1] - pos2.x[1]) * (pos1.x[1] - pos2.x[1]));
}

void Engine2D::FindPair() {
    particlePairs_.resize(0);
    for (int64_t bucketY = 0; bucketY < BUCKET_NUM_Y; ++bucketY) {
        for (int64_t bucketX = 0; bucketX < BUCKET_NUM_X; ++bucketX) {
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
                    auto tgtIdx = tgtY * BUCKET_NUM_X + tgtX;
                    auto curIdx = bucketY * BUCKET_NUM_X + bucketX;
                    for (const auto& cur: idxBucket_[curIdx]) {
                        for (const auto& tgt: idxBucket_[tgtIdx]) {
                            if (cur == tgt) {
                                continue;
                            }
                            if (Distance(pos_[cur], pos_[tgt]) < SMOOTHING_LENGTH) {
                                particlePairs_.emplace_back(cur, tgt);
                            }
                        }
                    }
                }
            }
        }
    }
}

void Engine2D::StepOne() {
    FindPair();
    UpdatePosVelocity();
    time_ += DT;
}
void Engine2D::UpdatePosVelocity() {
    for (uint64_t i = 0; i < pos_.size(); ++i) {
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
            idxBucket_[origBucket].erase(i);
            idxBucket_[newBucket].emplace(i);
        }
        u_[i].x[0] += (f_[i].x[0] + G_FORCE.x[0]) / PARTICLE_MASS * DT;
        u_[i].x[1] += (f_[i].x[1] + G_FORCE.x[1]) / PARTICLE_MASS * DT;
    }
}
}// namespace Sph