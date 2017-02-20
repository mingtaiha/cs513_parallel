#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include "cuda_runtime.h"
#include <thrust/scan.h>

using namespace std;

#define TPB 1024
#define RANGE 100

#define min(a,b) ((a < b) ? a : b)



__global__
void scat_part_sum(double * array, double * array_psums) {

	// Distributes the values from array_psums (array of partial sums) to every element
	// in the array. Every thread in a block gets the same partial sum added to it
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	//__syncthreads();

	array[tid] += array_psums[blockIdx.x];
	//__syncthreads();
}

__global__
void upsweep (double * array, double * array_aggr1, int size) {

	// Performs an upsweep

	int bid = blockIdx.x * blockDim.x;
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	int min_size = min(size, TPB);
	__syncthreads();

	// Merge elements like a binary tree
	
	for (int step = 2; step <= min_size ; step *= 2) {
		if (tid % step == (step - 1) && (tid - (step / 2) >= bid)) {
			array[tid] += array[tid - (step / 2)];
		}
		__syncthreads();
	}
	__syncthreads();

	// Aggregates the sum of each block to another array for to calculate partial tums

	if (array_aggr1 != NULL) {
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
}

__global__ 
void incl_downsweep (double * array) {
	
	int next_bid = (blockIdx.x + 1) * blockDim.x;
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	__syncthreads();

	// Performs an inclusive down sweep. After the inclusive down sweep, each block
	// will have elements a_1, a_1 + a_2, ... , a_1 + a_2 + ... + a_1024

	for (int step = TPB / 2; step > 1; step /= 2) {
		if (tid % step == (step - 1) && (tid + (step / 2) < next_bid)) {
			array[tid + (step / 2)] += array[tid];
		}
		__syncthreads();
	}
}

__global__
void excl_downsweep (double * array, int size) {

	int bsize = blockIdx.x * blockDim.x;
	int next_block = (blockIdx.x + 1) * blockDim.x;
	int tid = bsize + threadIdx.x;
	int tmp;

	int min_size = min(size, TPB);

	// Performs an exlusive down sweep. After the inclusive down sweep, each block
	// will have elements 0, 0 + a_1 , 0 + a_1 + a_2, ... , 0 + a_1 + a_2 + ... + a_1023

	if (tid % TPB == 0) { array[min(size, next_block) - 1] = 0; } 
	__syncthreads();

	for (int step = min_size; step > 0; step /= 2) {
		if (tid % step == (step - 1) && (tid - (step / 2) >= bsize)) {
			tmp = array[tid];
			array[tid] += array[tid - (step / 2)];
			array[tid - (step / 2)] = tmp;
		}
		__syncthreads();
	}
}

int main(int argc, char** argv)
{
	if (argc != 2) {
		cout << "Takes one argument - the number of elements in an array" << endl;
		return 0;
	}

	int size = atoi(argv[1]);
	int size_div1 = int(ceil(double(size) / double(TPB)));
	int size_div2 = int(ceil(double(size_div1) / double(TPB)));
	int nblocks = int(ceil(double(size) / double(TPB)));
	int nblocks_div1 = int(ceil(double(nblocks) / double(TPB)));
	int nblocks_div2 = int(ceil(double(nblocks_div1) / double(TPB)));

	cout << "First stage blocks: " << nblocks << endl;
	cout << "Second stage blocks: " << nblocks_div1 << endl;
	cout << "Third stage blocks: " << nblocks_div2 << endl;
	cout << "First stage size: " << size << endl;
	cout << "Second stage size: " << size_div1 << endl;
	cout << "Third stage size: " << size_div2 << endl;

	cout << "min(size_div1, TPB): " << min(size_div1, TPB) << endl;

	cout << "Malloc'ing\n";
	double *x = (double*)malloc(size * sizeof(double));
	double *x1 = (double*)malloc(size_div1 * sizeof(double));
	double *x2 = (double*)malloc(size_div2 * sizeof(double));
	double *y = (double*)malloc(size * sizeof(double));
	
	double *d_x, *d_x1, *d_x2;
	cudaMalloc(&d_x, size * sizeof(double));
	cudaMalloc(&d_x1, size_div1 * sizeof(double));
	cudaMalloc(&d_x2, size_div2 * sizeof(double));

	cout << "Generating Array\n";
	srand(time(NULL));
	for (int i = 0; i < size; i++) {
		x[i] = rand() % RANGE;
		y[i] = x[i];
	}

	
	clock_t seq_start, par_start;
	clock_t seq_pass, par_pass;

	cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x1, x1, size_div1 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x2, x2, size_div2 * sizeof(double), cudaMemcpyHostToDevice);


	for (int pf_step = 1; pf_step < 3; pf_step++) {
	cout << "Prefix Sum, iteration " << pf_step << "\n\n";

	cout << "Sequential Version" << endl;
	seq_start = clock();
	for (int i = 1; i < size; i++) {
		y[i] = y[i] + y[i - 1];
	}
	seq_pass = clock() - seq_start;
	cout << "Time taken (ms): " << seq_pass * 1000 / CLOCKS_PER_SEC << endl;

	cout << "Parallel Version" << endl;
	par_start = clock();
//	cout << "Up-Sweep\n" << endl;
	upsweep <<<nblocks, TPB>>> (d_x, d_x1, size);
	cudaDeviceSynchronize();

//	cout << "Up-Sweep 2\n" << endl;
	upsweep <<<nblocks_div1, TPB>>> (d_x1, d_x2, size_div1);
	cudaDeviceSynchronize();

//	cout << "Up-Sweep 3\n" << endl;
	upsweep <<<nblocks_div1, TPB>>> (d_x2, NULL, size_div2);
	cudaDeviceSynchronize();

//	cout << "Down-Sweep 3\n" << endl;
	excl_downsweep <<<nblocks_div2, TPB>>> (d_x2, size_div2);
	cudaDeviceSynchronize();

/*
	cudaMemcpy(x2, d_x2, size_div2 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < size_div2; i++) {
		cout << i << " " << std::setprecision(10) << x2[i] << endl;
	}
*/
//	cout << "Down-Sweep 2\n" << endl;
	excl_downsweep <<<nblocks_div1, TPB>>> (d_x1, size_div1);
	cudaDeviceSynchronize();
/*
	cudaMemcpy(x1, d_x1, size_div1 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < size_div1; i++) {
		cout << i << " " << std::setprecision(10) << x1[i] << endl;
	}
*/
//	cout << "Down-Sweep\n" << endl;
	incl_downsweep <<<nblocks, TPB>>> (d_x);
	cudaDeviceSynchronize();

//	cout << "Scatter Partial Sums 2\n" << endl;
	scat_part_sum <<<nblocks_div1, TPB>>> (d_x1, d_x2);
	cudaDeviceSynchronize();
/*
	cudaMemcpy(x1, d_x1, size_div1 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int i = 0; i < size_div1; i++) {
		cout << i << " " << std::setprecision(10) << x1[i] << endl;
	}
*/
//	cout << "Scatter Partial Sums 1\n" << endl;
	scat_part_sum <<<nblocks, TPB>>> (d_x, d_x1);
	cudaDeviceSynchronize();
	par_pass = clock() - par_start;
	cout << "Time taken (ms): " << par_pass * 1000 / CLOCKS_PER_SEC << endl;

/*	
	thrust::inclusive_scan(x, x + size, x);
	cudaDeviceSynchronize();
	thrust::inclusive_scan(x, x + size, x);
	cudaDeviceSynchronize();
*/

	cout << "Comparing Values" << endl;

	cudaMemcpy(x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
	//cout << size-1 << " " << std::setprecision(10) << x[size-1] << " " << y[size-1] << endl;

	for (int i = 0; i < size; i++) {
		//cout << i << " " << x[i] << " " << y[i] << endl;
		if (x[i] != y[i]) {
			cout << "Not the same at index " << i << endl;
			cout << i << " " << x[i] << " " << y[i] << endl;
			if (pf_step == 1) { break; }
			else { return 0;}

		}
	}

	cout << "arrays are the same\n\n\n";

	}



	return 0;
}
