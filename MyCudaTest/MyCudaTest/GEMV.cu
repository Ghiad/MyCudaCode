#include "GEMV.cuh"

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>((&pointer))[0])
#define FINAL_MASK 0xffffffff

using namespace std; 
const int thread_per_block = 256;

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
	if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
	if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
	if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
	if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
	if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
	return sum;
}

// if N == 32
__global__ void Sgemv_v0(
	float* __restrict__ A,
	float* __restrict__ x,
	float* __restrict__ y,
	const int M,
	const int N) {
	// Block index
	int bx = blockIdx.x;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int warp_size = 32;
	int laneId = tx % warp_size;
	int current_row = blockDim.y * bx + ty;

	if (current_row < M) {
		float res = 0;
		int kIteration = N / warp_size;
		if (kIteration == 0) kIteration = 1;
#pragma unroll
		for (int i = 0; i < kIteration; i++) {
			int current_col = i * warp_size + laneId;
			res += A[current_row * N + current_col] * x[current_col];
		}
		res = warpReduceSum<warp_size>(res);
		if (laneId == 0) y[current_row] = res;
	}
}

__global__ void Sgemv_v1(
	float* __restrict__ A,
	float* __restrict__ x,
	float* __restrict__ y,
	const int M,
	const int N) {
	// Block index
	int bx = blockIdx.x;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int warp_size = 32;
	int laneId = tx % warp_size;
	int current_row = blockDim.y * bx + ty;

	if (current_row < M) {
		float res = 0;
		int kIteration = (N / warp_size) / 4;
		if (kIteration == 0) kIteration = 1;
		A = &A[current_row * N];
#pragma unroll
		for (int i = 0; i < kIteration; i++) {
			int current_col_vec = (i * warp_size + laneId);
			float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
			float4 current_x = reinterpret_cast<float4*>(x)[current_col_vec];
			res += current_val.x * current_x.x;
			res += current_val.y * current_x.y;
			res += current_val.z * current_x.z;
			res += current_val.w * current_x.w;
		}
		res = warpReduceSum<warp_size>(res);
		if (laneId == 0) y[current_row] = res;
	}
}

template <
	const int ROW_PER_WARP
>
__global__ void Sgemv_v2(
	float* __restrict__ A,
	float* __restrict__ x,
	float* __restrict__ y,
	const int M,
	const int N) {
	// Block index
	int bx = blockIdx.x;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int warp_size = 32;
	int laneId = tx % warp_size;
	int current_warp_row = (blockDim.y * bx + ty) * ROW_PER_WARP;
	const int kWarp_size = warp_size / ROW_PER_WARP;
	int kLaneId = laneId % kWarp_size;
	int current_thread_row = current_warp_row + laneId / kWarp_size;

	if (current_thread_row < M) {
		float res = 0;
		int current_col = kLaneId;
		res += A[current_thread_row * N + current_col] * x[current_col];
		res = warpReduceSum<kWarp_size>(res);
		if (kLaneId == 0) y[current_thread_row] = res;
	}
}

__global__ void gemv_32(float* A, float* X, float* Y, int m, int n) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int warp_size = 32;
	int start_row = blockIdx.x * blockDim.y + ty;
	int start_col = tx % warp_size;
	float accum = 0.0f;
	if (start_row < m) {
		int loop = n / warp_size;
		if (loop == 0) loop = 1;
		for (int i = 0; i < loop; i++) {
			accum += A[start_row*n+ i * warpSize + start_col] * X[start_col];
		}
	}
	for (int i = 16; i > 0; i >>= 1) {
		accum += __shfl_xor_sync(FINAL_MASK,accum, i, 32);
	}
	if (start_col == 0) Y[start_row] = accum;
}

__global__ void gemv_128(float* A, float* X, float* Y, int m, int n) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int warp_size = 32;
	int start_row = blockIdx.x * blockDim.y + ty;
	int start_col = tx % warp_size;
	float accum = 0.0f;
	if (start_row < m) {
		int loop = (n / warp_size)/4;
		if (loop == 0) loop = 1;
		A = &A[start_row * n];
		for (int i = 0; i < loop; i++) {
			int current_col_vec = (i * warp_size + start_col);
			float4 current_val = reinterpret_cast<float4*>(A)[current_col_vec];
			float4 current_x = reinterpret_cast<float4*>(X)[current_col_vec];
			accum += current_val.x * current_x.x;
			accum += current_val.y * current_x.y;
			accum += current_val.z * current_x.z;
			accum += current_val.w * current_x.w;

		}
	}
	for (int i = 16; i > 0; i >>= 1) {
		accum += __shfl_xor_sync(FINAL_MASK, accum, i, 32);
	}
	if (start_col == 0) Y[start_row] = accum;
}
template<const int ROW_PER_WARP>
__global__ void gemv_16 (float* A, float* X, float* Y, int m, int n) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;

	int warp_size = 32;
	int lane_id = tx % warp_size;
	int current_warp_row = (bx * blockDim.y + ty) * ROW_PER_WARP;
	int kWarp_size = warp_size / ROW_PER_WARP;
	int curren_thread_row = current_warp_row  + lane_id / kWarp_size;
	int kLane_id = lane_id % kWarp_size;
	float accum = 0.0f;

	if (curren_thread_row < m) {
		accum += A[curren_thread_row * n + kLane_id] * X[kLane_id];
		for (int i = 8; i > 0; i >>= 1) {
			accum += __shfl_xor_sync(FINAL_MASK, accum, i, 32);
		}
		//accum = warpReduceSum<kWarp_size>(res);
		if (kLane_id == 0) Y[curren_thread_row] = accum;
	}
	
}

void GEMV() {
	const int m = 1024, n = 16;
	
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
	int iter = 100000;
	float mseconds=0.0f;
	double msecPerMatrixMul[2] = { 0, 0 };
	double gigaFlops[2] = { 0, 0 };
	double flopsPerMatrixMul = 2.0 * m * n;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, size_Y, cudaMemcpyHostToDevice);

	const int row_per_warp = 2;
	const int thread_per_block = 128;
	const int warp_size = 32;
	const int warp_per_block = thread_per_block / warp_size;
	const int row_per_block = warp_per_block * row_per_warp;
	cudaEventRecord(start);
	
	for (int i = 0; i < iter; i++) {
		dim3 grid(m / row_per_block);
		dim3 block(32, warp_per_block);
		gemv_16<row_per_warp><< <grid, block >> > (d_A, d_X, d_Y, m, n);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&mseconds, start, stop);
	msecPerMatrixMul[0] = mseconds / iter;
	gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
	printf("My gemv_16 Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
		gigaFlops[0],
		msecPerMatrixMul[0],
		flopsPerMatrixMul);
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
	cudaEventElapsedTime(&mseconds, start, stop);

	cudaMemcpy(Y1, d_Y, size_Y, cudaMemcpyDeviceToHost);
	msecPerMatrixMul[1] = mseconds / iter;
	gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
	printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
		gigaFlops[1],
		msecPerMatrixMul[1],
		flopsPerMatrixMul);
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
	printf("%s GEMV per:%f%%\n", correct ? "Result= PASS" : "Result= FAIL", gigaFlops[0] / gigaFlops[1] * 100);
}