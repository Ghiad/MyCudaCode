#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>
#include<iostream>
#include <malloc.h>

template <
	const int BLOCK_SIZE_M,
	const int BLOCK_SIZE_N,
	const int BLOCK_SIZE_K
>
__global__ void Sgemm2(float* A, float* B, float* C, const int M, const int N, const int K);
void invokSgemm2();

