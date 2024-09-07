//
// Created by QIAQIA on 2024/9/5.
//

#ifndef TUTORIALS_ENGINE_H
#define TUTORIALS_ENGINE_H
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "thread_pool.h"

namespace Sph {
template<int Dim>
struct Pos {
    float x[Dim]{};
};
class Engine2D {
public:
    explicit Engine2D(int particleNum);
    ~Engine2D() = default;
    void Step();
    void StepOne();
    void UpdatePosVelocity();
    void FindPair();
    void FindPairPerBlock(uint64_t idx, uint64_t size);
    void VerifyPair();
    void UpdateDensity();
    void UpdateDensityPerBlock(uint64_t idx, uint64_t size);
    void UpdatePressure();
    void UpdatePressurePerBlock(uint64_t idx, uint64_t size);
    void UpdateForce();
    void UpdateForcePerBlock(uint64_t idx, uint64_t size);
    const std::vector<float>& GetXyzs();
    const std::vector<float>& GetColor();

    std::vector<Pos<2>> pos_{};
    std::vector<Pos<2>> u_{};
    std::vector<Pos<2>> f_{};
    std::vector<float> p_{};
    std::vector<float> rho_{};
    std::vector<float> xyzsVec_{};
    std::vector<float> colorVec_{};
    std::vector<uint64_t> occupiedBucket_{};
    std::vector<std::unordered_set<uint64_t>> idxBucket_{};
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> particlePairs_{};
    std::vector<std::pair<uint64_t, uint64_t>> particlePairsSingle_{};
    std::vector<std::vector<float>> pairDistance_{};
    std::vector<float> pairDistanceSingle_{};
    ThreadPool pool_{std::thread::hardware_concurrency()};

    float time_{0};

    constexpr static uint64_t RENDER_INTERVAL = 6;
    constexpr static float PARTICLE_MASS = 1;
    constexpr static float ISOTROPIC_EXPONENT = 30;
    constexpr static float BASE_DENSITY = 0.025;
    constexpr static uint64_t SMOOTHING_LENGTH = 5;
    constexpr static float DYNAMIC_VISCOSITY = 0.1;
    constexpr static float DAMPING_COEFFICIENT = -0.3;
    constexpr static Pos<2> G_FORCE = {0, -0.1};
    constexpr static float DT = 0.01;
    constexpr static uint64_t DOMAIN_WIDTH = 240;
    constexpr static uint64_t DOMAIN_HEIGHT = 160;
    constexpr static uint64_t BUCKET_NUM_X = DOMAIN_WIDTH / SMOOTHING_LENGTH;
    constexpr static uint64_t BUCKET_NUM_Y = DOMAIN_HEIGHT / SMOOTHING_LENGTH;
    constexpr static float MAX_ACC = 100;

    constexpr static float DOMAIN_X_LIM[2] = {
            SMOOTHING_LENGTH,
            DOMAIN_WIDTH - SMOOTHING_LENGTH,
    };
    constexpr static float DOMAIN_Y_LIM[2] = {
            SMOOTHING_LENGTH,
            DOMAIN_HEIGHT - SMOOTHING_LENGTH,
    };
    constexpr static float NORMALIZATION_DENSITY =
            (315 * PARTICLE_MASS) /
            (64 * M_PI * SMOOTHING_LENGTH * SMOOTHING_LENGTH * SMOOTHING_LENGTH * SMOOTHING_LENGTH * SMOOTHING_LENGTH *
             SMOOTHING_LENGTH * SMOOTHING_LENGTH * SMOOTHING_LENGTH * SMOOTHING_LENGTH);
    constexpr static float NORMALIZATION_PRESSURE_FORCE =
            -(45 * PARTICLE_MASS) / (M_PI * SMOOTHING_LENGTH * SMOOTHING_LENGTH * SMOOTHING_LENGTH * SMOOTHING_LENGTH *
                                     SMOOTHING_LENGTH * SMOOTHING_LENGTH);
    constexpr static float NORMALIZATION_VISCOUS_FORCE = -NORMALIZATION_PRESSURE_FORCE * DYNAMIC_VISCOSITY;
};

}// namespace Sph

#endif//TUTORIALS_ENGINE_H
