#include "NavieSgemm.cuh"
__global__ void navieSgemm(float* A, float* B, float* C, const int M, const int N, const int K) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	float accum = 0.0;
	for (int i = 0; i < K; i++) {
		accum += A[x * K + i] * B[y + i * N];
	}
	C[x * N + y] = accum;
}

void invokNavieSgemm() {
	const int m = 2048;
	const int n = 2048;
	const int k = 2048;

	size_t bytes_A = m * k * sizeof(float);
	size_t bytes_B = k * n * sizeof(float);
	size_t bytes_C = m * n * sizeof(float);
	float* A = (float*)malloc(bytes_A);
	float* B = (float*)malloc(bytes_B);
	float* C = (float*)malloc(bytes_C);
	float* C1 = (float*)malloc(bytes_C);

	float* dA, * dB, * dC;
	cudaMalloc((void**)&dA, bytes_A);
	cudaMalloc((void**)&dB, bytes_B);
	cudaMalloc((void**)&dC, bytes_C);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			A[i * k + j] = 1.0;
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < n; j++) {
			B[i * k + j] = 1.0;
		}
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			C[i * k + j] = -1.0;
		}
	}

	//测量参数配置
	double msecPerMatrixMul[2] = { 0, 0 };
	double gigaFlops[2] = { 0, 0 };
	double flopsPerMatrixMul = 2.0 * m * n * k;

	//核函数参数配置
	const int BLOCK_SIZE_M = 32;
	const int BLOCK_SIZE_N = 32;
	const int BLOCK_SIZE_K = 8;
	const int THREAD_SIZE_Y = 1;
	const int THREAD_SIZE_X = 1;
	//dim3 gridsize(m / 128, n / 128) 不能写死啊
	dim3 gridsize(m / BLOCK_SIZE_M, n / BLOCK_SIZE_N);
	dim3 blocksize(BLOCK_SIZE_M / THREAD_SIZE_Y, BLOCK_SIZE_N / THREAD_SIZE_X);

	//拷贝数据到device
	cudaMemcpy(dA, A, bytes_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, bytes_B, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, bytes_C, cudaMemcpyHostToDevice);

	//运行核函数
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float msecTotal = 0;
	int nIter = 100;
	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		navieSgemm<< <gridsize, blocksize >> > (dA, dB, dC, m, n, k);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	cudaMemcpy(C, dC, bytes_C, cudaMemcpyDeviceToHost);

	msecPerMatrixMul[0] = msecTotal / nIter;
	gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
	printf("My NavieSgemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
		gigaFlops[0],
		msecPerMatrixMul[0],
		flopsPerMatrixMul);

	//运行cublas
	cublasHandle_t blas_handle;
	cublasCreate(&blas_handle);
	float alpha = 1.0;
	float beta = 0;
	cudaMemcpy(dC, C, bytes_C, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	for (int run = 0; run < nIter; run++) {
		cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
			m, n, k, &alpha,
			dA, k, dB, n, &beta, dC, n
		);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);
	cudaMemcpy(C1, dC, bytes_C, cudaMemcpyDeviceToHost);

	msecPerMatrixMul[1] = msecTotal / nIter;
	gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
	printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
		gigaFlops[1],
		msecPerMatrixMul[1],
		flopsPerMatrixMul);
	cublasDestroy(blas_handle);

	//检测运行结果的正确性
	double eps = 1.e-6;  // machine zero
	bool correct = true;
	for (int i = 0; i < m * n; i++) {
		int row = i / n;
		int col = i % n;
		double abs_err = fabs(C[i] - C1[col * m + row]);
		double dot_length = m;
		double abs_val = fabs(C[i]);
		double rel_err = abs_err / abs_val / dot_length;
		if (rel_err > eps) {
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
				i, C[i], C1[col * m + row], eps);
			correct = false;
			break;
		}
	}
	printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
	printf("ratio= %f %%\n", gigaFlops[0] / gigaFlops[1] *100);

	//释放内存
	free(A);
	free(B);
	free(C);
	free(C1);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}