# Smooth Particle Hydrodynamic Simulator with OpenGL based rendering & OpenCL based calculation.

1. Install dependencies:
    ```shell
   vcpkg install opencl
   vcpkg install opengl
    ```
2. Modify header file if needed. Change:
    ```c++
    #include <CL/cl.hpp>
    ```
    into:
    
    ```c++
    #include <CL/cl2.hpp>
    ```
3. By default, the particle color is determined by density. To produce the result in the sample, please change the <<*particles.cl*>> shader file. Change:
   ```opencl
    colorVec[i * 4 + 0] = diff < 1 ? 1 : 2 - diff;
    colorVec[i * 4 + 1] = diff < 1 ? diff : 1;
    colorVec[i * 4 + 2] = 0;
    colorVec[i * 4 + 3] = 1;
   ```
    into:
    ```opencl
    colorVec[i * 4 + 0] = i < 4000;
    colorVec[i * 4 + 1] = 4000 <= i && i < 6000;
    colorVec[i * 4 + 2] = 6000 <= i;
    colorVec[i * 4 + 3] = 1;
    ```
4. Run the program in the <<**PROJECT_ROOT_DIRECTORY**>> (the one contains shader/ folder).
5. It shall produce the following result:

<img src="https://github.com/ShunlongHu/SPH_Simulator/blob/master/SPH.gif" width="150%" height="150%"/>
