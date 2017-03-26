#include "genresult.cuh"
#include "cuda.h"
#include <sys/time.h>

__global__ void getMulAtomic_kernel(const int nnz, const int * coord_row, const int * coord_col, const float * A, const float * x, float * y){
    /* This is just an example empty kernel, you don't have to use the same kernel
 * name*/

        int thread_id = threadIdx.x;
        int thread_num = blockDim.x * gridDim.x;
        int iter = nnz % thread_num ? nnz/thread_num + 1: nnz/thread_num;

        for (int i = 0; i < iter; i++)
        {
            int dataid = thread_id + i * thread_num;
            if (dataid < nnz) {
                float data = A[dataid];
                int row = coord_row[dataid];
                int col = coord_col[dataid];
                float tmp = data * x[col];
                atomicAdd(&y[row], tmp);
            }
        }
}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*Allocate here...*/
   
/*
    printf("Printing Matrix information\n");
    printf("\tM x N: %d x %d\n", mat->M, mat->N);
    printf("\tnonzero terms: %d\n", mat->nz);
    printf("\trIndex[1]: %d\n", mat->rIndex[1]);
    printf("\tcIndex[1]: %d\n", mat->cIndex[1]);
    printf("\tval[1]: %f\n", mat->val[0]);

    printf("Printing Vector information\n");
    printf("\tM x N: %d x %d\n", vec->M, vec->N);
    printf("\tnonzero terms: %d\n", vec->nz);
    //printf("\trIndex[1]: %d\n", vec->rIndex[1]);
    //printf("\tcIndex[1]: %d\n", vec->cIndex[1]);
    printf("\tval[1]: %f\n", vec->val[1]);
*/

    //  Allocating space for Matrix mat on GPU
    int * d_rIndex, * d_cIndex;
    float * d_val;
    cudaMalloc(&d_rIndex, mat->nz * sizeof(int));
    cudaMalloc(&d_cIndex, mat->nz * sizeof(int));
    cudaMalloc(&d_val, mat->nz * sizeof(float));
    cudaMemcpy(d_rIndex, mat->rIndex, mat->nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cIndex, mat->cIndex, mat->nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, mat->val, mat->nz * sizeof(float), cudaMemcpyHostToDevice);

    //  Allocating space for Vector vec on GPU
    float * d_vec;
    cudaMalloc(&d_vec, vec->nz * sizeof(float));
    cudaMemcpy(&d_vec, vec->val, mat->nz * sizeof(float), cudaMemcpyHostToDevice);

    // Allocating space for Vector out_vec on Host and GPU
    float * out_vec, * d_out_vec;
    out_vec = (float *) malloc(mat->M * sizeof(float));
    cudaMalloc(&d_out_vec, mat->M * sizeof(float));

	/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernels...*/

    getMulAtomic_kernel<<<blockNum, blockSize>>>(mat->nz, d_rIndex, d_cIndex, d_val, d_vec, d_out_vec);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    /*please modify the timing to milli-seconds*/
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
    /*Deallocate.*/

    cudaMemcpy(out_vec, d_out_vec, mat->M * sizeof(float), cudaMemcpyDeviceToHost);

    free(out_vec);
    cudaFree(d_rIndex);
    cudaFree(d_cIndex);
    cudaFree(d_val);
    cudaFree(d_out_vec);
}
