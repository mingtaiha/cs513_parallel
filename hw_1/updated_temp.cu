#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD 1024
#define POWER 10

using namespace std;

// Kernel function to add the elements of two arrays
__global__
void sums(int n, float *x, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i == 0) {
            x[i] = x[i];
        }
        else if ((i+1) % offset == 0) {
            x[i] = x[i] + x[i-1];
        }
        else {
            printf("%d inside else\n", i);
            x[i] = x[i];
        }
    }
}

__global__
void upsweep(int n, float *x, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();

    if (i < n) {
    for (; offset <= n; offset *= 2) {
        if ((i+1) % offset == 0) {
            x[i] += x[i - (offset / 2)];
        }
        __syncthreads();
    }
   }
    __syncthreads();
}

__global__
void downsweep(int n, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();

    if (i < n) {
    for (int offset = 1 << 25; offset > 1; offset /= 2) {
        if ((i + 1) % offset == 0) {
            x[i + (offset / 2)] += x[i];
        }
        __syncthreads();
    }}
    __syncthreads();
}

int main (void) {
    // int n = 1 << POWER;
    //int n = 1 << 12;
    int n = 450000000;
    int offset = 2;

    float *x, *d_x;

    x = (float*)malloc(n * sizeof(float));

    cudaMalloc(&d_x, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        x[i] = (float)(i % 10);
    }

    clock_t start, end;
    int diff;
    start = clock();

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

    upsweep <<< (n + THREAD - 1) / THREAD, THREAD>>> (n, d_x, offset);
    cudaDeviceSynchronize();

    downsweep <<< (n + THREAD - 1) / THREAD, THREAD>>> (n, d_x);
    cudaDeviceSynchronize();

    upsweep <<< (n + THREAD - 1) / THREAD, THREAD>>> (n, d_x, offset);
    cudaDeviceSynchronize();

    downsweep <<< (n + THREAD - 1) / THREAD, THREAD>>> (n, d_x);
    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

    end = clock();
    
    diff = (end - start) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken to run sequential algorithm: %d msec\n", diff);



    // for (int i = 0; i < n; i++) {
    //    // cout << i << " " << x[i] << endl;
    //    cout << x[i] << endl;
    // }
    return 0;
}

