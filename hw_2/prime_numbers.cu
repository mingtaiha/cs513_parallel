#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

#define min(a, b) ((a <= b) ? a : b)


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
			return;
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
void par_sieve(int * d_arr, int n, int sqrt_n, int start, int end) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int min_i = min(start, 2);
	int max_i = min(sqrt_n, end);

	__syncthreads();


	// Performs Sieve of Eratosthenese
	// Go from i = 2, ... , sqrt_n
	for (int i = min_i; i <= max_i; i++) {
		// Only uses sqrt_n threads (to minimize using sqrt(n) processors
		if (tid < sqrt_n) {
			// Checks if marked as 1 (prime)
			if (d_arr[i-2] == 1) {
				// Perform interleaved work. With sqrt_n processors, each processor
				// goes in increments of i, starting from 2i and not exceeding n.
				// This traversal ensures that every thread will set to 0 a number
				// which is a multiple of i.

				// This implementation does not introduce more work than the original
				// implementation. So, there is O(n*log(log(n))) work. With sqrt_n
				// processors, I can use O(sqrt_n * log(log(n)))
				for (int j = 0; ((j + tid + 1) * i + (i-2)) < (n-1); j+=sqrt_n) {
					d_arr[(j + tid + 1) * i + (i - 2)] = 0;
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
    cout << "sequential implementation\n";
	int * seq_array = make_array_2_to_n(n);
	//print_array(seq_array, n);
	
	// Sequential Sieve
    clock_t start, seq_runtime;
    start = clock();
	seq_sieve(seq_array, n);
    seq_runtime = clock() - start;
    cout << "sequential run time (in milliseconds): " << (seq_runtime * 1000 / CLOCKS_PER_SEC) << endl;
	//print_prime(seq_array, n);
    cout << "\n\n";

	cout << "parallel\n";
	// Initializing variables for parallel execution
	int sqrt_n = int(ceil(sqrt(int(n))));
	int * par_array = make_array_2_to_n(n);
	int * d_par_array;

	cout << "cudaMalloc\n";
	cudaError_t malloc_error = cudaMalloc((void**)&d_par_array, sizeof(int) * (n-1));
	if (malloc_error != cudaSuccess) {
		printf("cudaMalloc error: %s\n", cudaGetErrorString(malloc_error));
	}

	cout << "cudaMemcpyHostToDevice\n";

    clock_t with_memcpy_start, with_memcpy_runtime;
	with_memcpy_start = clock();

    cudaError_t memcpy_to_d_error = cudaMemcpy((void*)d_par_array, (void*)par_array, sizeof(int) * (n-1), cudaMemcpyHostToDevice);
	if (malloc_error != cudaSuccess) {
		printf("cudaMemcpyHostToDevice: %s\n", cudaGetErrorString(memcpy_to_d_error));
	}

	// Defining threads per block (tpb), and number of blocks to schedule
	int tpb = 1024;
	int nblocks = n / tpb + 1;
	
    clock_t without_memcpy_start, without_memcpy_runtime;
	without_memcpy_start = clock();
	// Calling Parallel Sieve
    if (n <= 20000000) {
        cout << "Kernel call 1" << endl;
	    par_sieve<<<nblocks, tpb>>>(d_par_array, n, sqrt_n, 2, sqrt_n);
	    cudaDeviceSynchronize();
    } else if (n <= 27500000) {
        cout << "Kernel call 1" << endl;
	    par_sieve<<<nblocks, tpb>>>(d_par_array, n, sqrt_n, 2, 1000);
	    cudaDeviceSynchronize();
        cout << "Kernel call 2" << endl;
	    par_sieve<<<nblocks, tpb>>>(d_par_array, n, sqrt_n, 1001, 3000);
	    cudaDeviceSynchronize();
        cout << "Kernel call 3" << endl;
	    par_sieve<<<nblocks, tpb>>>(d_par_array, n, sqrt_n, 3001, sqrt_n);
	    cudaDeviceSynchronize();
    } else {
        cout << "I have not been able to get n > 27 500 000 to run without without CUDA launch timeout" << endl;
        cout << "Exiting now\n";
        return 0;
    }

	// Error checking
	cudaError_t kernel_error = cudaGetLastError();
	if (kernel_error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(kernel_error));
	}

    without_memcpy_runtime = clock() - without_memcpy_start;
    cout << "parallel run time (in milliseconds) WITHOUT cudaMemcpy: " << (without_memcpy_runtime * 1000 / CLOCKS_PER_SEC) << endl;
	

	cudaMemcpy((void*)par_array, (void*)d_par_array, sizeof(int) * (n-1), cudaMemcpyDeviceToHost);
    
    with_memcpy_runtime = clock() - with_memcpy_start;
    cout << "parallel run time (in milliseconds) WITH cudaMemcpy: " << (with_memcpy_runtime * 1000 / CLOCKS_PER_SEC) << endl;
	//print_prime(par_array, n);

	diff_prime(seq_array, par_array, n);

	return 0;

}
