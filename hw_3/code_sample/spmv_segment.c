#include "genresult.cuh"
#include <sys/time.h>

#define BLOCK_SIZE 1024

__device__ void segmented_scan(const int lane, const int * rows, float * vals, float * y) {

    for (int i = 1; i < 32; i *= 2) {
        if (lane >= i && rows[threadIdx.x] == rows[threadIdx.x - i]) {
            vals[threadIdx.x] += vals[threadIdx.x - i];
        }
    }
    if (lane == 31 || (lane < 31 && rows[threadIdx.x] != rows[threadIdx.x + 1])) {
        atomicAdd(&y[rows[threadIdx.x]], vals[threadIdx.x]);
    }
}


__global__ void segmented_spmv_kernel(const int nnz, const int * rows, const int * cols, const float * A, const float * x, float * y){
    /*Put your kernel(s) implementation here, you don't have to use exactly the
 * same kernel name */

    __shared__ float v[BLOCK_SIZE];
    __shared__ int r[BLOCK_SIZE];

    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int thread_num = blockDim.x * gridDim.x;
    int iter = nnz % thread_num ? nnz/thread_num + 1 : nnz/thread_num;
  
    for (int i = 0; i < iter; i++) {
        int prodid = thread_id + i * thread_num;
        if (prodid < nnz) {
            v[threadIdx.x] = A[prodid] * x[cols[prodid]];
            r[threadIdx.x] = rows[prodid];
            segmented_scan(thread_id % 32, r, v, y);
        }
    }
}
/*
typedef struct A_cont {
    int r;
    int c;
    float v;
} A_cont;
*/

int comparator(const void *a_i, const void *a_j) {

    A_cont * a_i_cast = (A_cont *) a_i;
    A_cont * a_j_cast = (A_cont *) a_j;
    
    return a_i_cast->r - a_j_cast->r;
}


void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){


    // Preprocessing
    
    A_cont * tmpA = (A_cont *) malloc(sizeof(A_cont) * mat->nz);
    
    for (int i = 0; i < mat->nz; i++) {
        tmpA[i].r = mat->rIndex[i];
        tmpA[i].c = mat->cIndex[i];
        tmpA[i].v = mat->val[i];
    }

    qsort(tmpA, mat->nz, sizeof(A_cont), comparator);

    for (int j = 0; j < mat->nz; j++) {
        mat->rIndex[j] = tmpA[j].r;
        mat->cIndex[j] = tmpA[j].c;
        mat->val[j] = tmpA[j].v;
    }


    /*Allocate things...*/


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
    cudaMemcpy(d_vec, vec->val, vec->nz * sizeof(float), cudaMemcpyHostToDevice);

    // Allocating space for Vector out_vec on Host and GPU
    float * out_vec, * d_out_vec;
    out_vec = (float *) calloc(mat->M, sizeof(float));
    cudaMalloc(&d_out_vec, mat->M * sizeof(float));
    cudaMemcpy(d_out_vec, out_vec, mat->M * sizeof(float), cudaMemcpyHostToDevice);

	/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernel(s)*/

    segmented_spmv_kernel<<<blockNum, blockSize>>>(mat->nz, d_rIndex, d_cIndex, d_val, d_vec, d_out_vec);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
          printf("CUDA error: %s\n", cudaGetErrorString(error));
          exit(-1);
    }


    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    /*Deallocate, please*/
    cudaMemcpy(res->val, d_out_vec, mat->M * sizeof(float), cudaMemcpyDeviceToHost);

    free(out_vec);
    cudaFree(d_rIndex);
    cudaFree(d_cIndex);
    cudaFree(d_val);
    cudaFree(d_out_vec);
}
