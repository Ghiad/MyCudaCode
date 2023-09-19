#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>
#include<iostream>
#include <malloc.h>

template <
	const int BLOCK,
	const int STRIDE
>
__global__ void Sgemm3(float* a, float* b, float* c, const int m, const int n, const int k);
void invokSgemm3();

