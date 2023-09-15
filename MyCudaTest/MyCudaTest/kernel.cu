#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include<iostream>
#include <malloc.h>

void printDeviceProp(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cout << "Using GPU device " << dev << ": " << deviceProp.name << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / 1024 / 1024 / 1024 << " GB" << std::endl;
    std::cout << "Number of Multi-Processors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << "Max Block Size: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    std::cout << "Clock Rate: " << deviceProp.clockRate << " kHz" << std::endl;
    std::cout << "Memory Clock Rate: " << deviceProp.memoryClockRate << " kHz" << std::endl;
    std::cout << "Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
    std::cout << std::endl;
}
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
int main() {
    
    float* a = (float*)malloc(4 * sizeof(float));
    float* b = (float*)malloc(4 * sizeof(float));
    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 4.0;
    a[4] = 5.0;
    a[5] = 6.0;
    a[6] = 7.0;
    a[7] = 8.0;
    FETCH_FLOAT4(b[0]) = FETCH_FLOAT4(a[0]);
    std::cout << b[0];

    return 0;
}