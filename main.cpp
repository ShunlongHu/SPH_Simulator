#include <iostream>

#include "render.h"
#include "window.h"

using namespace Sph;
int main() {
    Window window;
    Render::Init();
    window.Run([]() { Render::Step(); });
    return 0;
}

//#include <CL/cl.hpp>
//#include <iostream>
//#include <string>
//#include <vector>
//
//static std::string opencl_kernel =
//        R"(
//    __kernel void vecadd
//    (
//        __global int *A,
//        __global int *B,
//        __global int *C
//    )
//    {
//        int id = get_global_id(0);
//        C[id] = A[id] + B[id];
//    }
//)";
//
//int main(int argc, char **argv) {
//
//    std::vector<cl::Platform> platforms;
//    cl::Platform::get(&platforms);
//
//    if (platforms.empty()) {
//        std::cerr << "No platforms!" << std::endl;
//        return -1;
//    }
//
//    cl::Platform platform = platforms[0];
//    std::vector<cl::Device> Devices;
//
//    platform.getDevices(CL_DEVICE_TYPE_GPU, &Devices);
//    if (Devices.empty()) {
//        std::cerr << "No Devices!" << std::endl;
//        return -1;
//    }
//
//    cl::Device device = Devices[0];
//    std::cout << "Device : " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
//
//    cl::Context context({device});
//    cl::Program program(context, opencl_kernel);
//
//    if (program.build() != CL_SUCCESS) {
//        std::cerr << "Fail to build" << std::endl;
//        return -1;
//    }
//
//    std::vector<int> A(10000000, 1);
//    std::vector<int> B(10000000, 2);
//    std::vector<int> C(10000000, 0);
//
//    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 100);
//    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 100);
//    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 100);
//
//    cl::CommandQueue queue(context, device);
//
//    cl::Kernel vec_add(program, "vecadd");
//
//    vec_add.setArg(0, buffer_A);
//    vec_add.setArg(1, buffer_B);
//    vec_add.setArg(2, buffer_C);
//
//    queue.enqueueWriteBuffer(buffer_A, CL_FALSE, 0, sizeof(int) * 100, A.data());
//    queue.enqueueWriteBuffer(buffer_B, CL_FALSE, 0, sizeof(int) * 100, B.data());
//
//    queue.enqueueNDRangeKernel(vec_add, cl::NullRange, cl::NDRange(100), cl::NullRange);
//
//    queue.enqueueReadBuffer(buffer_C, CL_FALSE, 0, sizeof(int) * 100, C.data());
//
//    queue.finish();
//
//    for (const auto &v: A) {
//        std::cout << v << " ";
//    }
//    std::cout << std::endl;
//
//    for (const auto &v: B) {
//        std::cout << v << " ";
//    }
//    std::cout << std::endl;
//
//    for (const auto &v: C) {
//        std::cout << v << " ";
//    }
//    std::cout << std::endl;
//
//    return 0;
//}