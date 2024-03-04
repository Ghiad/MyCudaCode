#include "reduce.cuh"

using namespace std;

const int thread_per_block = 256;
#define FINAL_MASK 0xffffffff

#define checkCudaError(fun) \
{ \
	cudaError_t e = (fun); \
	if (e != cudaSuccess) \
		printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));\
}

__device__ void warpReduce(volatile float* s_data, int tid) {
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid + 8];
	s_data[tid] += s_data[tid + 4];
	s_data[tid] += s_data[tid + 2];
	s_data[tid] += s_data[tid + 1];
}

__global__ void reduceSum1(float* in, float* out) {
	int tid = threadIdx.x;
	__shared__ float s_in[thread_per_block];
	s_in[tid] = in[blockIdx.x * blockDim.x + tid];
	__syncthreads();
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			s_in[tid] += s_in[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) {
		out[blockIdx.x] = s_in[0];
	}
}

__global__ void reduceSum2(float* in, float* out) {
	int tid = threadIdx.x;
	__shared__ float s_data[thread_per_block];
	s_data[tid] = in[blockIdx.x * blockDim.x + tid];
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			s_data[index] += s_data[index + s];
		}
		__syncthreads();
	}
	if (tid == 0) out[blockIdx.x] = s_data[0];
}

__global__ void reduceSum3(float* in, float* out) {
	int tid = threadIdx.x;
	__shared__ float s_data[thread_per_block];
	s_data[tid] = in[blockIdx.x * blockDim.x + tid];
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 1; s >>= 1) {
		if (tid < s) {
			s_data[tid] += s_data[tid + s];
		}

		__syncthreads();
	}
	if (tid == 0) out[blockIdx.x] = s_data[0];
}

template<int num_per_block>
__global__ void reduceSum4(float* in, float* out) {
	int tid = threadIdx.x;
	__shared__ float s_data[thread_per_block];
	float acc = 0.0f;
	for (int i = 0; i < num_per_block; i += thread_per_block) {
		acc += in[blockIdx.x * num_per_block + tid + i];
	}
	s_data[tid] = acc;
	__syncthreads();

	for (int s = blockDim.x / 2; s >= 1; s >>= 1) {
		if (tid < s) {
			s_data[tid] += s_data[tid + s];
		}

		__syncthreads();
	}
	if (tid == 0) {
		out[blockIdx.x] = s_data[0];
		//printf("%d %f \n", blockIdx.x, s_data[0]);
	}
}

template<int num_per_block>
__global__ void reduceSum5(float* in, float* out) {
	int tid = threadIdx.x;
	__shared__ float s_data[thread_per_block];
	float acc = 0.0f;
	for (int i = 0; i < num_per_block; i += thread_per_block) {
		acc += in[blockIdx.x * num_per_block + tid + i];
	}
	s_data[tid] = acc;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			s_data[tid] += s_data[tid + s];
		}

		__syncthreads();
	}
	if (tid < 32) warpReduce(s_data, tid);

	if (tid == 0) out[blockIdx.x] = s_data[0];

}

template<int num_per_block>
__global__ void reduceSum6(float* in, float* out) {
	int tid = threadIdx.x;
	__shared__ float s_data[thread_per_block];
	float acc = 0.0f;
	for (int i = 0; i < num_per_block; i += thread_per_block) {
		acc += in[blockIdx.x * num_per_block + tid + i];
	}
	s_data[tid] = acc;
	__syncthreads();

	int blockSize = blockDim.x;
	if (blockSize >= 512) {
		if (tid < 256) {
			s_data[tid] += s_data[tid + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			s_data[tid] += s_data[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			s_data[tid] += s_data[tid + 64];
		}
		__syncthreads();
	}

	if (tid < 32) warpReduce(s_data, tid);

	if (tid == 0) out[blockIdx.x] = s_data[0];

}

template<int num_per_block>
__global__ void reduceSum7(float* in, float* val) {
	int tid = threadIdx.x;
	float acc = 0.0f;
	for (int i = 0; i < num_per_block; i += thread_per_block) {
		acc += in[blockIdx.x * num_per_block + tid + i];
	}
	__syncthreads();

	int warpId = tid / 32;
	int laneId = tid % 32;
	__shared__ float sum[32];
	for (int i = 16; i >= 1; i >>= 1) {
		acc += __shfl_xor_sync(FINAL_MASK, acc, i, 32);
	}
	if (laneId == 0) sum[warpId] = acc;
	__syncthreads();
	acc = (tid < blockDim.x / 32) ? sum[laneId] : 0.0f;

	for (int i = 16; i >= 1; i >>= 1) {
		acc += __shfl_xor_sync(FINAL_MASK, acc, i, 32);
	}
	if (tid == 0) val[blockIdx.x] = acc;
}

template<const int num_per_block>
__global__ void reduceSumTest(float* in, float* val) {
	int tx = threadIdx.x;

	__shared__ float s[thread_per_block];
	float accum = 0.0f;
	for (int i = tx; i < num_per_block; i += blockDim.x) {
		accum += in[i + blockIdx.x * num_per_block];
	}
	s[tx] = accum;
	__syncthreads();

	for (int i = thread_per_block / 2; i > 0; i >>= 1) {
		if (tx < i) {
			s[tx] += s[i + tx];
		}
		__syncthreads();
	}
	if (tx == 0) val[blockIdx.x] = s[0];
}
bool check(float* a, float* b, int size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) return false;
	}
	return true;
}

void reduce() {
	const int size = 32 * 1024 * 1024;
	//const int block_num = size / thread_per_block;
	const int block_num = 1024;
	const int num_per_block = size / block_num;
	//float* a = (float*)malloc(sizeof(float) * size);
	float* a = new float[size];
	float* b = (float*)malloc(sizeof(float) * block_num);
	float* c = (float*)malloc(sizeof(float) * block_num);
	float* d_a, * d_b;
	for (int i = 0; i < size; i++) {
		a[i] = 1.0;
		if (i < block_num) {
			b[i] = -1.0;
			c[i] = 0.0;
		}
	}

	/*for (int i = 0; i < block_num; i++) {
		float cur = 0.0;
		for (int j = 0; j < thread_per_block; j++) {
			cur += a[i * thread_per_block + j];
		}
		c[i] = cur;
	}*/
	for (int i = 0; i < block_num; i++) {
		float cur = 0.0;
		for (int j = 0; j < num_per_block; j++) {
			cur += a[i * num_per_block + j];
		}
		c[i] = cur;
	}


	cudaMalloc((void**)&d_a, size * sizeof(float));
	cudaMalloc((void**)&d_b, size * sizeof(float));

	cudaMemcpy(d_a, a, size * sizeof(float), cudaMemcpyHostToDevice);


	float stime = 0.0f;
	int iter = 1000;
	dim3 girdsize(block_num, 1);
	dim3 blocksize(thread_per_block, 1);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for (int i = 0; i < iter; i++) {
		reduceSum7<num_per_block> << < girdsize, blocksize >> > (d_a, d_b);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&stime, start, stop);
	stime /= iter;
	cudaMemcpy(b, d_b, block_num * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	if (!check(b, c, block_num)) printf("error");
	else {
		printf("success time: %f", stime * 1000 * 1000);
	}

}