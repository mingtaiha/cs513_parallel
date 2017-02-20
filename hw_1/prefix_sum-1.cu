#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD 1024

using namespace std;

__global__
void upsweep(int n, double *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();

    for (int offset = 2; offset <= n; offset *= 2) {
        if ((i+1) % offset == 0) {
            x[i] += x[i - (offset / 2)];
        }
        __syncthreads();
    }
    __syncthreads();
}

__global__
void downsweep(int n, double *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();

    for (int offset = 1 << 25; offset > 1; offset /= 2) {
        if ((i + 1) % offset == 0) {
            x[i + (offset / 2)] += x[i];
        }
        __syncthreads();
    }
    __syncthreads();
}

void validate(int n, double *x, double *y){
    // printf("inside validate\n");
    // for (int i = 0; i < n; i++) {
    //     printf("%d\n", y[i]);
    // }
    for (int i = 1; i < n; i++) {
        y[i] += y[i-1];
    }

    for (int i = 1; i < n; i++) {
        y[i] += y[i-1];
    }

    // printf("--------\n");
    // for (int i = 0; i < n; i++) {
    //    cout << y[i] << endl;
    // }

    int correct = 0;
    int incorrect = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] == y[i]) {
            correct++;
        }
        else {
            incorrect++;
        }
    }
    printf("Number correct: %d\n",correct);
    printf("Number incorrect: %d\n",incorrect);
}

int main(int argc, const char * argv[]) {

    if (argc != 2) {
        printf("The argument is wrong! Execute your program with only input file name!\n");
        return 1;
    }

    //Dummy code for creating a random input vectors
    //Convenient for the text purpose
    //Please comment out when you submit your code!!!!!!!!!
    /*FILE *fp = fopen(argv[1], "w");
    if (fp == NULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    int n = 1 << 24;
    fprintf(fp, "%d\n", n);
    srand(time(NULL));
    for (int i=0; i<n; i++)
        fprintf(fp, "%lg\n", ((double)(rand() % n))/100);
    fclose(fp);
    printf("Finished writing\n");*/

    //Read input from input file specified by user
    FILE* fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("The file can not be opened or does not exist!\n");
        return 1;
    }
    int n;
    fscanf(fp, "%d\n", &n);
    printf("%d\n", n);

    // sum(a, b, n);

    //n = 4500;

    double *x, *d_x;

    x = (double*)malloc(n * sizeof(double));

    cudaMalloc(&d_x, n * sizeof(double));

    double *seq;
    seq = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        fscanf(fp, "%lg\n", &x[i]);
    }
    memcpy(seq, x, n * sizeof(double));



/*for (int i = 0; i < n; i++) {
    x[i] = (double)(i % 10);
}
*/
    fclose(fp);

    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    upsweep <<< (n + THREAD - 1) / THREAD, THREAD>>> (n, d_x);
    cudaDeviceSynchronize();

    downsweep <<< (n + THREAD - 1) / THREAD, THREAD>>> (n, d_x);
    cudaDeviceSynchronize();

    upsweep <<< (n + THREAD - 1) / THREAD, THREAD>>> (n, d_x);
    cudaDeviceSynchronize();

    downsweep <<< (n + THREAD - 1) / THREAD, THREAD>>> (n, d_x);
    cudaDeviceSynchronize();

    cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < n; i++) {
    //    cout << x[i] << endl;
    // }

    validate(n, x, seq);

        //Write b into output file
    fp = fopen("output.txt","w");
    if (fp == NULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    fprintf(fp, "%d\n", n);
    for (int i=0; i<n; i++)
        fprintf(fp, "%lg\n", x[i]);
    fclose(fp);

    cudaFree(d_x);
    free(x);
        // free(b);
    printf("Done...\n");
    return 0;
}

