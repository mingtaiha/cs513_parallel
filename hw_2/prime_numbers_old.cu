#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>

using namespace std;




int * make_array_2_to_n(int n) {

	// Makes array of size n-1 (index 0 to n-2 map to 2 to n)
	int * array = (int *) malloc((n-1) * sizeof(int));
	for (int i = 0; i < (n-1); i++) {
		array[i] = 1;
	}
	return array;
}

void print_array(int * arr, int n) {

	for (int i = 0; i < (n-1); i++) {
		cout << (i+2) << " " << arr[i] << endl;
	}

}

void print_prime(int * arr, int n) {

	// if arr[i] == 1, then i+2 is prime (note the +2 shift
	//    because of the way I defined my matrix
	for (int i = 0; i < (n-1); i++) {
		if (arr[i] == 1) {
			cout << (i+2) << endl;
		}
	}
}

void diff_prime(int * arr1, int * arr2, int n) {

	// Checks if two arrays have the same input and output
	// Checks if both arrays are correct (or incorrect)
	int flag = 1;
	for (int i = 0; i < (n-1); i++) {
		if (arr1[i] != arr2[i]) {
			if (flag == 1) { flag = 0; }
			cout << "Arrays are different\n";
			cout << (i+2) << " " << arr1[i] << " " << arr2[i] << endl;
		}
	}
	if (flag == 1) {
		cout << "Arrays are the same\n";
	}
}	


void seq_sieve(int * arr, int n) {

	int sqrt_n = int(ceil(sqrt(int(n))));
	int i_sqr;	

	// Sieve of Eratosthenese
	for (int i = 2; i <= sqrt_n; i++) {
		if (arr[i-2] == 1) {
			i_sqr = i * i;
			for (int j = i_sqr; j <= n; j+=i) {
				arr[j - 2] = 0;
			}
		}
	}
}

__global__
void par_sieve(int * d_arr, int n, int sqrt_n) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	__syncthreads();

	// Performs Sieve of Eratosthenese
	// Go from i = 2, ... , sqrt_n
	for (int i = 2; i <= sqrt_n; i++) {
		// Only uses sqrt_n threads (to minimize using sqrt(n) processors
		if (tid <= sqrt_n) {
			// Checks if marked as 1 (prime)
			if (d_arr[i-2] == 1) {
				// Perform interleaved work. With sqrt_n subarrays of sqrt_n elements
				// each, every thread (of sqrt_n threads total) will check one element in
				// the subarray and mark as 0 (composite) if the element is a multiple
				// of i (2, ..., sqrt_n).
				for (int j = 0; j < n; j+=sqrt_n) {
					if ((j + tid + (2*i) - 2 < n) && (((j + tid + i) % i) == 0)) {
						d_arr[j + tid + (2*i) - 2] = 0;
					}
				}
			}
		}
	}

}

int main(int argc, char** argv) {

	if (argc != 2) {
		cout << "Takes one argument - n, positive integer - to calculate the number of primes at most n\n";
	}

	int n = atoi(argv[1]);

	// Making Array
	int * seq_array = make_array_2_to_n(n);
	//print_array(seq_array, n);
	
	// Sequential Sieve
	seq_sieve(seq_array, n);
	//print_prime(seq_array, n);


	// Initializing variables for parallel execution
	int sqrt_n = int(ceil(sqrt(int(n))));
	int * par_array = make_array_2_to_n(n);
	int * d_par_array;

	cudaMalloc((void**)&d_par_array, sizeof(int) * (n-1));
	cudaMemcpy((void*)d_par_array, (void*)par_array, sizeof(int) * (n-1), cudaMemcpyHostToDevice);

	// Defining threads per block (tpb), and number of blocks to schedule
	int tpb = 1024;
	int nblocks = n / tpb + 1;
	
	cout << "parallel \n\n\n";

	// Calling Parallel Sieve
	par_sieve<<<nblocks, tpb>>>(d_par_array, n, sqrt_n);
	cudaDeviceSynchronize();

	// Error checking
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}
	

	cudaMemcpy((void*)par_array, (void*)d_par_array, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
	//print_prime(par_array, n);

	//diff_prime(seq_array, par_array, n);

	return 0;

}
