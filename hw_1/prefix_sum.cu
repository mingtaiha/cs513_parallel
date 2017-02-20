#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "cuda_runtime.h"

using namespace std;

#define TPB 1024
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



void sum(double* a, double* b, const int n) {
    //Given an array a[0...n-1], you need to compute b[0...n-1],
    //where b[i] = (i+1)*a[0] + i*a[1] + ... + 2*a[i-1] + a[i]
    //Note that b is NOT initialized with 0, be careful!
    //Write your CUDA code starting from here
    //Add any functions (e.g., device function) you want within this file

        int size = n;
        int size_div1 = int(ceil(double(size) / double(TPB)));
        int size_div2 = int(ceil(double(size_div1) / double(TPB)));
        int nblocks = int(ceil(double(size) / double(TPB)));
        int nblocks_div1 = int(ceil(double(nblocks) / double(TPB)));
        int nblocks_div2 = int(ceil(double(nblocks_div1) / double(TPB)));

        double *d_x, *d_x1, *d_x2;
        cudaMalloc(&d_x, size * sizeof(double));
        cudaMalloc(&d_x1, size_div1 * sizeof(double));
        cudaMalloc(&d_x2, size_div2 * sizeof(double));

        cudaMemcpy(d_x, a, size * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_x1, x1, size_div1 * sizeof(double), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_x2, x2, size_div2 * sizeof(double), cudaMemcpyHostToDevice);

        for (int pf_step = 1; pf_step < 3; pf_step++) {

//      cout << "Up-Sweep\n" << endl;
        upsweep <<<nblocks, TPB>>> (d_x, d_x1, size);
        cudaDeviceSynchronize();

//      cout << "Up-Sweep 2\n" << endl;
        upsweep <<<nblocks_div1, TPB>>> (d_x1, d_x2, size_div1);
        cudaDeviceSynchronize();

//      cout << "Up-Sweep 3\n" << endl;
        upsweep <<<nblocks_div1, TPB>>> (d_x2, NULL, size_div2);
        cudaDeviceSynchronize();

//      cout << "Down-Sweep 3\n" << endl;
        excl_downsweep <<<nblocks_div2, TPB>>> (d_x2, size_div2);
        cudaDeviceSynchronize();

//      cout << "Down-Sweep 2\n" << endl;
        excl_downsweep <<<nblocks_div1, TPB>>> (d_x1, size_div1);
        cudaDeviceSynchronize();

//      cout << "Down-Sweep\n" << endl;
        incl_downsweep <<<nblocks, TPB>>> (d_x);
        cudaDeviceSynchronize();

//      cout << "Scatter Partial Sums 2\n" << endl;
        scat_part_sum <<<nblocks_div1, TPB>>> (d_x1, d_x2);
        cudaDeviceSynchronize();

//      cout << "Scatter Partial Sums 1\n" << endl;
        scat_part_sum <<<nblocks, TPB>>> (d_x, d_x1);
        cudaDeviceSynchronize();
    
        cudaMemcpy(b, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
	}

}

int main(int argc, const char * argv[]) {

    if (argc != 2) {
        printf("The argument is wrong! Execute your program with only input file name!\n");
        return 1;
    }
    
	
    int n = 1 << 24;
    //Dummy code for creating a random input vectors
    //Convenient for the text purpose
    //Please comment out when you submit your code!!!!!!!!! 	
/*    FILE *fpw = fopen(argv[1], "w");
    if (fpw == NULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    //int n = 1 << 24;
    fprintf(fpw, "%d\n", n);
    srand(time(NULL));
    for (int i=0; i<n; i++)
        fprintf(fpw, "%lg\n", ((double)(rand() % n))/100);
    fclose(fpw);
    printf("Finished writing\n");
*/    
    //Read input from input file specified by user
    FILE* fpr = fopen(argv[1], "r");
    if (fpr == NULL) {
        printf("The file can not be opened or does not exist!\n");
        return 1;
    }
    //int n;
    fscanf(fpr, "%d\n", &n);
    printf("%d\n", n);
    double* a = (double*)malloc(n*sizeof(double));
    double* b = (double*)malloc(n*sizeof(double));
    for (int i=0; i<n; i++) {
        fscanf(fpr, "%lg\n", &a[i]);
    }
    fclose(fpr);
    
    //Main function
    sum(a, b, n);
    
    //Write b into output file
    FILE* fpo = fopen("output.txt","w");
    if (fpo == NULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    fprintf(fpo, "%d\n", n);
    for (int i=0; i<n; i++)
        fprintf(fpo, "%lg\n", b[i]);
    fclose(fpo);
    free(a);
    free(b);
    printf("Done...\n");
    return 0;
}
