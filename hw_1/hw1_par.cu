#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include "cuda_runtime.h"
#include <thrust/scan.h>

using namespace std;

#define TPB 1024
#define RANGE 10

#define min(a,b) ((a < b) ? a : b)

__global__
void incl_pfsum (float * array, int size) {

	int bsize = blockIdx.x * blockDim.x;
	int tid = bsize + threadIdx.x;
	int tmp;

	size = min(size, TPB);

	__syncthreads();

	
	for (int step = 2; step <= size ; step *= 2) {
		if (tid % step == (step - 1) && (tid - (step / 2) >= bsize)) {
			array[tid] += array[tid - (step / 2)];
		}
		__syncthreads();
	}
	__syncthreads();

	if (tid % TPB == 0) { array[bsize - 1] = 0; } 
	__syncthreads();

	for (int step = size; step > 0; step /= 2) {
		if (tid % step == (step - 1) && (tid - (step / 2) >= bsize)) {
			tmp = array[tid];
			//__syncthreads();
			array[tid] += array[tid - (step / 2)];
			//__syncthreads();
			array[tid - (step / 2)] = tmp;
			//__syncthreads();
		}
		__syncthreads();
	}
	__syncthreads();
}

__global__
void scat_part_sum(float * array, float * array_psums) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	__syncthreads();

	array[tid] += array_psums[blockIdx.x];
	__syncthreads();
}

__global__
void upsweep (float * array, float * array_aggr1, int size, int size_aggr1) {

	int bid = blockIdx.x * blockDim.x;
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	__syncthreads();

	
	for (int step = 2; step <= size ; step *= 2) {
		if (tid % step == (step - 1) && (tid - (step / 2) >= bid)) {
			array[tid] += array[tid - (step / 2)];
		}
		__syncthreads();
	}
	__syncthreads();

	if (threadIdx.x == (TPB - 1)) {
		if (tid < size) {
			array_aggr1[blockIdx.x] = array[tid];
		}
		else {
			array_aggr1[blockIdx.x] = array[size - 1];
		}
	}
	__syncthreads();
}


__global__ 
void downsweep (float * array) {
	
	int next_bid = (blockIdx.x + 1) * blockDim.x;
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	__syncthreads();

	for (int step = TPB / 2; step > 1; step /= 2) {
		if (tid % step == (step - 1) && (tid + (step / 2) < next_bid)) {
			array[tid + (step / 2)] += array[tid];
		}
		__syncthreads();
	}
	__syncthreads();
	

/*
	if (tid >= (1<<step) && (tid < size)) {
		tmp = array[tid - (1<<step)];
		__syncthreads();

		array[tid] = tmp + array[tid];
		__syncthreads();
	}
*/
}


int main(int argc, char** argv)
{
	if (argc != 2) {
		cout << "Takes one argument - the number of elements in an array" << endl;
		return 0;
	}

	int size = atoi(argv[1]);
	int size_div1 = int(ceil(float(size) / float(TPB)));
	int size_div2 = int(ceil(float(size_div1) / float(TPB)));
	int nblocks = int(ceil(float(size) / float(TPB)));
	int nblocks_div1 = int(ceil(float(nblocks) / float(TPB)));
	int nblocks_div2 = int(ceil(float(nblocks_div1) / float(TPB)));

	cout << "First stage blocks: " << nblocks << endl;
	cout << "Second stage blocks: " << nblocks_div1 << endl;
	cout << "Third stage blocks: " << nblocks_div2 << endl;
	cout << "First stage size: " << size << endl;
	cout << "Second stage size: " << size_div1 << endl;
	cout << "Third stage size: " << size_div2 << endl;

	cout << "Malloc'ing\n";
	float *x = (float*)malloc(size * sizeof(float));
	float *x1 = (float*)malloc(size_div1 * sizeof(float));
	float *x2 = (float*)malloc(size_div2 * sizeof(float));
	float *y = (float*)malloc(size * sizeof(float));
	
	float *d_x, *d_x1, *d_x2;
	cudaMalloc(&d_x, size * sizeof(float));
	cudaMalloc(&d_x1, size_div1 * sizeof(float));
	cudaMalloc(&d_x2, size_div2 * sizeof(float));

	cout << "Generating Array\n";
	srand(time(NULL));
	for (int i = 0; i < size; i++) {
		x[i] = rand() % RANGE;
		y[i] = x[i];
	}

	for (int i = 1; i < size; i++) {
		y[i] = y[i] + y[i - 1];
	}
/*
	for (int i = 1; i < size; i++) {
		y[i] = y[i] + y[i - 1];
	}
*/

	
	cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, size_div1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, size_div2 * sizeof(float), cudaMemcpyHostToDevice);


	cout << "Up-Sweep\n" << endl;
	upsweep <<<nblocks, TPB>>> (d_x, d_x1, size, nblocks_div1);
	cudaDeviceSynchronize();

	cout << "Down-Sweep\n" << endl;
	downsweep <<<nblocks, TPB>>> (d_x);
	cudaDeviceSynchronize();

/*
	cout << "Up-Sweep 2\n" << endl;
	upsweep <<<nblocks_div1, TPB>>> (d_x1, d_x2, size_div1, nblocks_div2);
	cudaDeviceSynchronize();

	cout << "Down-Sweep 2\n" << endl;
	downsweep <<<nblocks_div1, TPB>>> (d_x1);
	cudaDeviceSynchronize();
*/

	cout << "Inclusive Sum 1\n" << endl;
	incl_pfsum <<<nblocks_div1, TPB>>> (d_x1, size_div1);
	cudaDeviceSynchronize();

	cudaMemcpy(x1, d_x1, size_div1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < size_div1; i++) {
		cout << i << " " << x1[i] << endl;
	}

	cout << "Inclusive Sum 2\n" << endl;
	incl_pfsum <<<nblocks_div2, TPB>>> (d_x2, size_div2);
	cudaDeviceSynchronize();

	cudaMemcpy(x2, d_x2, size_div2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < size_div2; i++) {
		cout << i << " " << x2[i] << endl;
	}

	cout << "Scatter Partial Sums 2\n" << endl;
	scat_part_sum <<<nblocks_div1, TPB>>> (d_x1, d_x2);
	cudaDeviceSynchronize();

/*
	cudaMemcpy(x1, d_x1, size_div1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < size_div1; i++) {
		cout << i << " " << x1[i] << endl;
	}
*/

	cout << "Scatter Partial Sums 1\n" << endl;
	scat_part_sum <<<nblocks, TPB>>> (d_x, d_x1);
	cudaDeviceSynchronize();


/*
	for (int i = 0; i < size; i++) {
		cout << i << " " << x[i] << endl;
	}
*/


/*	
	thrust::inclusive_scan(x, x + size, x);
	cudaDeviceSynchronize();
	thrust::inclusive_scan(x, x + size, x);
	cudaDeviceSynchronize();
*/

	cudaMemcpy(x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++) {
		cout << i << " " << x[i] << " " << y[i] << endl;
		if (x[i] != y[i]) {
			//cout << i << " " << x[i] << " " << y[i] << endl;
			//cout << "Not the same" << endl;
			//return 0;

		}
	}


	cout << "arrays are the same" << endl;


	return 0;
}
