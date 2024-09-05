//
// Created by QIAQIA on 2024/9/5.
//

#include "engine.h"

#include <iostream>
using namespace std;
namespace Sph {
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
        auto bucketX = static_cast<uint64_t>(x / SMOOTHING_LENGTH);
        auto bucketY = static_cast<uint64_t>(y / SMOOTHING_LENGTH);
        idxBucket_[bucketY * BUCKET_NUM_X + bucketX].emplace(i);
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
    for (int i = 0; i < RENDER_INTERVAL; ++i) { StepOne(); }
    cout << time_ << endl;
}
void Engine2D::StepOne() {
    for (uint64_t i = 0; i < pos_.size(); ++i) {
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


        u_[i].x[0] += (f_[i].x[0] + G_FORCE.x[0]) / PARTICLE_MASS * DT;
        u_[i].x[1] += (f_[i].x[1] + G_FORCE.x[1]) / PARTICLE_MASS * DT;
    }
    time_ += DT;
}
}// namespace Sph