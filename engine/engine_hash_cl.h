//
// Created by QIAQIA on 2024/9/5.
//

#ifndef TUTORIALS_ENGINE_HASH_CL_H
#define TUTORIALS_ENGINE_HASH_CL_H
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include "cl2.hpp"
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "thread_pool.h"

namespace Sph {
#ifndef POS_HEADER
#define POS_HEADER
template<int Dim>
struct Pos {
    float x[Dim]{};
};
#endif
class EngineHashCL2D {
public:
    explicit EngineHashCL2D(int particleNum);
    ~EngineHashCL2D() = default;
    void InitOpenCl(int particleNum);
    void Step();
    void StepOne();
    void UpdateBucket();
    void UpdateHashCL();
    void UpdateStartIdxCL();
    void UpdatePosVelocityCL();
    void UpdateDensityPressureCL();
    void UpdateForceCL();
    void BitonicMergeSortCL();

    // opencl
    const std::vector<float>& GetXyzs();
    const std::vector<float>& GetColor();

    std::vector<Pos<2>> pos_{};
    std::vector<float> xyzsVec_{};
    std::vector<float> colorVec_{};

    ThreadPool pool_{std::thread::hardware_concurrency()};

    float time_{0};
    constexpr static uint64_t RENDER_INTERVAL = 6;
    constexpr static float PARTICLE_MASS = 1;
    constexpr static float ISOTROPIC_EXPONENT = 200;
    constexpr static float BASE_DENSITY = 0.025;
    constexpr static float SMOOTHING_LENGTH = 5;
    constexpr static float DYNAMIC_VISCOSITY = 0.1;
    constexpr static float DAMPING_COEFFICIENT = -0.3;
    constexpr static Pos<2> G_FORCE = {0, -0.1};
    constexpr static float DT = 0.01;
    constexpr static float DOMAIN_WIDTH = 240;
    constexpr static float DOMAIN_HEIGHT = 160;
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


    // OpenCL members
    uint32_t particleNum_;
    uint32_t nextPowOf2_{2};
    cl::Context ctx_;
    cl::Program program_;
    std::vector<size_t> localWorkSize_;
    cl::CommandQueue q_;
    cl::Kernel hashKernel_;
    cl::Kernel startIdxKernel_;
    cl::Kernel densityPressureKernel_;
    cl::Kernel forceKernel_;
    cl::Kernel posVelocityKernel_;
    cl::Kernel sortStartKernel_;
    cl::Kernel sortLocalKernel_;
    cl::Kernel sortGlobalKernel_;
    cl::Kernel colorKernel_;
    cl::Kernel xyzsKernel_;
    cl::Buffer bucketInBuf_;// key, idx
    cl::Buffer bucketOutBuf_;
    cl::Buffer unsortedBucketBuf_;
    cl::Buffer bucketKeyStartIdxMapBuf_;
    cl::Buffer posBuf_;
    cl::Buffer uBuf_;
    cl::Buffer fBuf_;
    cl::Buffer pBuf_{};
    cl::Buffer rhoBuf_{};
    cl::Buffer xyzsBuf_{};
    cl::Buffer colorBuf_{};
};

}// namespace Sph

#endif//TUTORIALS_ENGINE_HASH_CL_H
