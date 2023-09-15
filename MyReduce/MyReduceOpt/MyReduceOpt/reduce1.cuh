#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>

const long int N = 4 * 1024 * 1024;
const int Num_per_block = 256;
const int Thread_per_block = 256;

__global__ void reduce1(float* d_in, float* d_out);

