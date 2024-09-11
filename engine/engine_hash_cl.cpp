//
// Created by QIAQIA on 2024/9/5.
//

#include "engine_hash_cl.h"

#include <algorithm>
#include <fstream>
#include <iostream>

#include "cl_utils.h"

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


EngineHashCL2D::EngineHashCL2D(int particleNum) : particleNum_(particleNum) {
    for (; nextPowOf2_ < particleNum_; nextPowOf2_ *= 2) {
    }
    nextPowOf2_ = max(2048U, nextPowOf2_);// otherwise sorting will be reversed
    srand(0);
    pos_.resize(particleNum);
    xyzsVec_.resize(particleNum * 4);
    colorVec_.resize(particleNum * 4);

    for (int i = 0; i < OBSTACLE_NUM; ++i) {
        auto diff = OBSTACLE_LEN / OBSTACLE_NUM;
        pos_[i] = {DOMAIN_X_LIM[0],
                   (DOMAIN_Y_LIM[1] - DOMAIN_Y_LIM[0]) / 2 + diff * i - OBSTACLE_LEN / 2 + DOMAIN_Y_LIM[0]};
    }

    auto freeParticleNum = particleNum - OBSTACLE_NUM;
    auto area = (DOMAIN_Y_LIM[1] - DOMAIN_Y_LIM[0]) * (DOMAIN_X_LIM[1] - DOMAIN_X_LIM[0]);
    auto particleDensity = freeParticleNum / area;
    auto spacing = sqrt(1 / particleDensity);
    auto xNum = static_cast<int>(ceil((DOMAIN_X_LIM[1] - DOMAIN_X_LIM[0]) / spacing));
    auto yNum = static_cast<int>(ceil((DOMAIN_Y_LIM[1] - DOMAIN_Y_LIM[0]) / spacing));
    auto xSpacing = (DOMAIN_X_LIM[1] - DOMAIN_X_LIM[0]) / xNum;
    auto ySpacing = (DOMAIN_Y_LIM[1] - DOMAIN_Y_LIM[0]) / yNum;

    for (int i = OBSTACLE_NUM; i < particleNum; ++i) {
        auto col = (i - OBSTACLE_NUM) % xNum;
        auto row = (i - OBSTACLE_NUM) / xNum;

        auto x = col * xSpacing + DOMAIN_X_LIM[0];
        auto y = row * ySpacing + DOMAIN_Y_LIM[0];
        pos_[i] = {x, y};
    }
    InitOpenCl(particleNum);
}

const std::vector<float>& EngineHashCL2D::GetColor() {
    colorKernel_.setArg(0, rhoBuf_);
    colorKernel_.setArg(1, colorBuf_);
    colorKernel_.setArg(2, BASE_DENSITY);
    q_.enqueueNDRangeKernel(colorKernel_, {}, {particleNum_}, {});
    q_.enqueueReadBuffer(colorBuf_, CL_TRUE, 0, sizeof(float) * particleNum_ * 4, colorVec_.data());
    return colorVec_;
}
const std::vector<float>& EngineHashCL2D::GetXyzs() {
    float scaling = max(DOMAIN_HEIGHT, DOMAIN_WIDTH) / 4.0f;
    float biasX = DOMAIN_WIDTH / -2.0f;
    float biasY = DOMAIN_HEIGHT / -2.0f;
    xyzsKernel_.setArg(0, posBuf_);
    xyzsKernel_.setArg(1, xyzsBuf_);
    xyzsKernel_.setArg(2, biasX);
    xyzsKernel_.setArg(3, biasY);
    xyzsKernel_.setArg(4, scaling);
    q_.enqueueNDRangeKernel(xyzsKernel_, {}, {particleNum_}, {});
    q_.enqueueReadBuffer(xyzsBuf_, CL_TRUE, 0, sizeof(float) * particleNum_ * 4, xyzsVec_.data());
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

inline void VerifySort(const vector<pair<uint32_t, uint32_t>>& value, const vector<uint32_t>& origValue) {
    for (uint32_t i = 1; i < value.size(); ++i) {
        if (value[i].second < value[i - 1].second) {
            cerr << "Sort Incorrect" << endl;
            exit(-1);
        }
    }
    for (uint32_t i = 0; i < value.size(); ++i) {
        if (value[i].second != origValue[value[i].first]) {
            cerr << "Sort Incorrect" << endl;
            exit(-1);
        }
    }
}

inline size_t GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize) {
    size_t r = DataElemCount % LocalWorkSize;
    if (r == 0) return DataElemCount;
    else
        return DataElemCount + LocalWorkSize - r;
}

void EngineHashCL2D::BitonicMergeSortCL() {
    size_t globalWorkSize[1];

    globalWorkSize[0] = GetGlobalWorkSize(nextPowOf2_ / 2, localWorkSize_[0]);
    unsigned int limit = (unsigned int) 2 * localWorkSize_[0];//limit is double the localWorkSize_

    // start with Sort_BitonicMergesortLocalBegin to sort local until we reach the limit
    sortStartKernel_.setArg(0, bucketInBuf_);
    sortStartKernel_.setArg(1, bucketOutBuf_);

    q_.enqueueNDRangeKernel(sortStartKernel_, {}, globalWorkSize[0], localWorkSize_[0], {});

    // proceed with global and local kernels
    for (unsigned int blocksize = limit; blocksize <= nextPowOf2_; blocksize <<= 1) {
        for (unsigned int stride = blocksize / 2; stride > 0; stride >>= 1) {
            if (stride >= limit) {
                //Sort_BitonicMergesortGlobal
                sortGlobalKernel_.setArg(0, bucketOutBuf_);
                sortGlobalKernel_.setArg(1, nextPowOf2_);
                sortGlobalKernel_.setArg(2, blocksize);
                sortGlobalKernel_.setArg(3, stride);

                q_.enqueueNDRangeKernel(sortGlobalKernel_, {}, globalWorkSize[0], localWorkSize_[0], {});
            } else {
                //Sort_BitonicMergesortLocal
                sortLocalKernel_.setArg(0, bucketOutBuf_);
                sortLocalKernel_.setArg(1, nextPowOf2_);
                sortLocalKernel_.setArg(2, blocksize);
                sortLocalKernel_.setArg(3, stride);

                q_.enqueueNDRangeKernel(sortLocalKernel_, {}, globalWorkSize[0], localWorkSize_[0], {});
            }
        }
    }
    swap(bucketInBuf_, bucketOutBuf_);
}

void EngineHashCL2D::UpdateBucket() {
    UpdateHashCL();
    BitonicMergeSortCL();
#ifdef VERIFY_SORT
    for (uint32_t idx = 0; idx < pos_.size(); ++idx) {
        auto key = CalcBucketHash(pos_[idx]);
        auto hash = key % pos_.size();
        unsortedBucket_[idx] = hash;
    }
    VerifySort(bucket_, unsortedBucket_);
#endif
    UpdateStartIdxCL();
}

void EngineHashCL2D::StepOne() {
    UpdateBucket();
    UpdateDensityPressureCL();
    UpdateForceCL();
    UpdatePosVelocityCL();
    time_ += DT;
}

void EngineHashCL2D::InitOpenCl(int particleNum) {
    // Read shader
    string fName = "shader/particles.cl";
    ifstream ifs(fName);
    if (!ifs.is_open()) {
        cerr << "Failed to open " << fName << endl;
    }
    ifs.seekg(0, ios_base::end);
    auto size = ifs.tellg();
    ifs.seekg(0, ios_base::beg);
    string prog(size, '\0');
    ifs.read(prog.data(), size);

    // Get local size
    cout << GetPlatformName(0) << endl;
    cout << GetDeviceName(0, 0) << endl;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No platforms!" << std::endl;
        exit(-1);
    }

    platform_ = platforms[0];
    std::vector<cl::Device> Devices;

    platform_.getDevices(CL_DEVICE_TYPE_GPU, &Devices);
    if (Devices.empty()) {
        std::cerr << "No Devices!" << std::endl;
        exit(-1);
    }

    device_ = Devices[0];
    std::cout << "Device : " << device_.getInfo<CL_DEVICE_NAME>() << std::endl;

    ctx_ = cl::Context({device_});
    localWorkSize_ = GetLocalWorkgroupSize(0, 0);

    // load prog
    prog = "#define MAX_LOCAL_SIZE " + to_string(localWorkSize_[0]) + '\n' + prog;
    cl_int ret;
    program_ = cl::Program(ctx_, prog, true, &ret);
    if (ret != CL_SUCCESS) {
        cout << getErrorString(ret) << endl;
        cout << GetBuildLog(program_, 0, 0) << endl;
        exit(-1);
    }

    // load kernel
    hashKernel_ = cl::Kernel(program_, "UpdateHashKernel");
    startIdxKernel_ = cl::Kernel(program_, "UpdateStartIdxKernel");
    densityPressureKernel_ = cl::Kernel(program_, "UpdateDensityPressureKernel");
    forceKernel_ = cl::Kernel(program_, "UpdateForceKernel");
    posVelocityKernel_ = cl::Kernel(program_, "UpdatePosVelocityKernel");
    sortStartKernel_ = cl::Kernel(program_, "Sort_BitonicMergesortStart");
    sortLocalKernel_ = cl::Kernel(program_, "Sort_BitonicMergesortLocal");
    sortGlobalKernel_ = cl::Kernel(program_, "Sort_BitonicMergesortGlobal");
    colorKernel_ = cl::Kernel(program_, "GetColorKernel");
    xyzsKernel_ = cl::Kernel(program_, "GetXyzsKernel");

    // load context
    q_ = cl::CommandQueue(ctx_, device_);

    // create buffer
    posBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, particleNum * sizeof(float) * 2);
    bucketInBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, nextPowOf2_ * sizeof(uint32_t) * 2);
    bucketOutBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, nextPowOf2_ * sizeof(uint32_t) * 2);
    bucketKeyStartIdxMapBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, particleNum_ * sizeof(uint32_t));
    rhoBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, particleNum_ * sizeof(float));
    pBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, particleNum_ * sizeof(float));
    uBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, particleNum * sizeof(float) * 2);
    fBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, particleNum * sizeof(float) * 2);
    xyzsBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, particleNum * sizeof(float) * 4);
    colorBuf_ = cl::Buffer(ctx_, CL_MEM_READ_WRITE, particleNum * sizeof(float) * 4);
}
void EngineHashCL2D::UpdateHashCL() {
    if (time_ == 0) {
        q_.enqueueWriteBuffer(posBuf_, CL_FALSE, 0, particleNum_ * sizeof(float) * 2, pos_.data());
    }
    hashKernel_.setArg(0, posBuf_);
    hashKernel_.setArg(1, bucketInBuf_);
    hashKernel_.setArg(2, particleNum_);
    hashKernel_.setArg(3, SMOOTHING_LENGTH);
    q_.enqueueNDRangeKernel(hashKernel_, {}, {nextPowOf2_}, {});
}
void EngineHashCL2D::UpdateStartIdxCL() {
    startIdxKernel_.setArg(0, bucketInBuf_);
    startIdxKernel_.setArg(1, bucketKeyStartIdxMapBuf_);
    q_.enqueueNDRangeKernel(startIdxKernel_, {}, {particleNum_}, {});
}
void EngineHashCL2D::UpdateDensityPressureCL() {
    densityPressureKernel_.setArg(0, bucketInBuf_);
    densityPressureKernel_.setArg(1, bucketKeyStartIdxMapBuf_);
    densityPressureKernel_.setArg(2, posBuf_);
    densityPressureKernel_.setArg(3, rhoBuf_);
    densityPressureKernel_.setArg(4, pBuf_);
    densityPressureKernel_.setArg(5, particleNum_);
    densityPressureKernel_.setArg(6, SMOOTHING_LENGTH);
    densityPressureKernel_.setArg(7, NORMALIZATION_DENSITY);
    densityPressureKernel_.setArg(8, ISOTROPIC_EXPONENT);
    densityPressureKernel_.setArg(9, BASE_DENSITY);
    q_.enqueueNDRangeKernel(densityPressureKernel_, {}, {particleNum_}, {});
}
void EngineHashCL2D::UpdateForceCL() {
    forceKernel_.setArg(0, bucketInBuf_);
    forceKernel_.setArg(1, bucketKeyStartIdxMapBuf_);
    forceKernel_.setArg(2, posBuf_);
    forceKernel_.setArg(3, rhoBuf_);
    forceKernel_.setArg(4, pBuf_);
    forceKernel_.setArg(5, uBuf_);
    forceKernel_.setArg(6, fBuf_);
    forceKernel_.setArg(7, particleNum_);
    forceKernel_.setArg(8, SMOOTHING_LENGTH);
    forceKernel_.setArg(9, NORMALIZATION_PRESSURE_FORCE);
    forceKernel_.setArg(10, NORMALIZATION_VISCOUS_FORCE);
    q_.enqueueNDRangeKernel(forceKernel_, {}, {particleNum_}, {});
}
void EngineHashCL2D::UpdatePosVelocityCL() {
    posVelocityKernel_.setArg(0, rhoBuf_);
    posVelocityKernel_.setArg(1, fBuf_);
    posVelocityKernel_.setArg(2, posBuf_);
    posVelocityKernel_.setArg(3, uBuf_);
    posVelocityKernel_.setArg(4, particleNum_);
    posVelocityKernel_.setArg(5, cl_float2{G_FORCE.x[0], G_FORCE.x[1]});
    posVelocityKernel_.setArg(6, cl_float2{DOMAIN_X_LIM[0], DOMAIN_X_LIM[1]});
    posVelocityKernel_.setArg(7, cl_float2{DOMAIN_Y_LIM[0], DOMAIN_Y_LIM[1]});
    posVelocityKernel_.setArg(8, DT);
    posVelocityKernel_.setArg(9, DAMPING_COEFFICIENT);
    posVelocityKernel_.setArg(10, MAX_ACC);
    posVelocityKernel_.setArg(11, OBSTACLE_NUM);
    posVelocityKernel_.setArg(12, OBSTACLE_V);
    q_.enqueueNDRangeKernel(posVelocityKernel_, {}, {particleNum_}, {});
}
}// namespace Sph