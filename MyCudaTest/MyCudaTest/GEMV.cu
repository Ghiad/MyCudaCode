#include "GEMV.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>((&pointer))[0])
#define FINAL_MASK 0xffffffff


using namespace std;
const int thread_per_block = 256;

__global__ void gemvKernel(float* A, float* X, float* Y, int m, int n) {
	int tid = threadIdx.x;
	int warp_id = tid / 32;
	int lane_id = tid % 32;
	int start_row = 8 * blockIdx.x + warp_id;
	int start_col = lane_id * 4;
	float4 a_reg = FETCH_FLOAT4(A[start_row * n + start_col]);
	float4 x_reg = FETCH_FLOAT4(X[start_col]);
	float y_reg = 0.0f;
	y_reg += a_reg.x * x_reg.x;
	y_reg += a_reg.y * x_reg.y;
	y_reg += a_reg.z * x_reg.z;
	y_reg += a_reg.w * x_reg.w;
	for (int mask = 16; mask >= 1; mask >>= 1) {
		y_reg += __shfl_xor_sync(FINAL_MASK, y_reg, mask, 32);

	}
	if (lane_id == 0) {
		Y[start_row] = y_reg;
		//printf("%f\n ", y_reg);
		//printf("%f\n", Y[start_row]);
	}

}

void GEMV() {
	const int m = 1024, n = 128;

	int size = m * n * sizeof(float);
	int size_X = n * sizeof(float);
	int size_Y = m * sizeof(float);
	float* A = (float*)malloc(size);
	float* Y = (float*)malloc(size_Y);
	float* X = (float*)malloc(size_X);
	float* Y1 = (float*)malloc(size_Y);

	float* d_A, * d_X, * d_Y;
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_Y, size_Y);
	cudaMalloc((void**)&d_X, size_X);

	for (int i = 0; i < m * n; i++) {
		A[i] = (float)i / n;
		//printf("%d %f\n", i, A[i]);
	}
	for (int i = 0; i < n; i++) {
		X[i] = 1;
	}
	memset(Y, 0, size_Y);
	memset(Y1, 0, size_Y);
	int iter = 100;
	float mseconds[2];
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, size_Y, cudaMemcpyHostToDevice);

	dim3 grid(m / (thread_per_block / 32));
	dim3 block(thread_per_block);
	cudaEventRecord(start);
	for (int i = 0; i < iter; i++) {
		gemvKernel << <grid, block >> > (d_A, d_X, d_Y, m, n);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mseconds[0], start, stop);
	cudaMemcpy(Y, d_Y, size_Y, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cublasHandle_t base_handle;
	cublasCreate(&base_handle);
	float alpa = 1.0;
	float beta = 0.0;
	cudaMemcpy(d_Y, Y1, size_Y, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	for (int i = 0; i < iter; i++) {
		cublasSgemv(base_handle, CUBLAS_OP_T,
			n, m, &alpa,
			d_A, n, d_X, 1, &beta, d_Y, 1);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mseconds[1], start, stop);
	cudaMemcpy(Y1, d_Y, size_Y, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cublasDestroy(base_handle);

	double eps = 1.e-6;  // machine zero
	bool correct = true;
	for (int i = 0; i < m; i++) {
		double abs_err = fabs(Y[i] - Y1[i]);
		double dot_length = m;
		double abs_val = fabs(Y[i]);
		double rel_err = abs_err / abs_val / dot_length;
		if (rel_err > eps) {
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
				i, Y[i], Y1[i], eps);
			correct = false;
			break;
		}
	}
	printf("%s per:%f%%\n", correct ? "Result= PASS" : "Result= FAIL", mseconds[1] / mseconds[0] * 100);
}