#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>
#include<iostream>
#include <malloc.h>

__global__ void Sgemm4(float* A, float* B, float* C, const int M, const int N, const int K);
void invokSgemm4();