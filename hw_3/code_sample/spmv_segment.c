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

/*
    // Iterating through all N elements, using all threads T, where T can be less than N
    for (int i = 1; i < iter + 1; i++)
    {
        // Storing products and row index in shared memory
        int prodid = thread_id + i * thread_num;
        if (prodid < nnz) {
            v[thread_id] = A[prodid] * x[cols[prodid]];
            r[thread_id] = rows[prodid];
        }
        __syncthreads();
        
        // First pass of segmented partial sum
        if (prodid < nnz) {
            for (int j = 1; j < blockDim.x; j = j*2) {
                if (thread_id >= j && r[thread_id] == r[thread_id - j]) {
                    v[thread_id] += v[thread_id - j];
                }
            }
        }
        __syncthreads();
        
        // Going through each thread and filtering out row indices where all products
        // in the row has been summed, and thus does not need to be considered further
        // This step has two purposes: 1) it filters output elements which do not need
        // to be reduced, 2) it guarantees that at most two elements will be written
        // to the row buffer and computation buffer. The second step makes the
        // aggregation of partial sums much more predictable
        if (thread_id == 0) {
            int first_row = r[0];               // First row encountered in shared memory
            int prev_row = r[0];                // Previous row encountered when scanning shared memory
            int cur_row = r[0];                 // Newest row encountered when scanning shared memory    
            int first_psum = 0;                 // First partial sum
            int last_psum = 0;                  // Last partial sum
            for (int j = 1; j < blockDim.x; j++) {
                if (r[j] != r[j-1]) {               // If rows change
                    prev_row = cur_row;             // Update previous and current row
                    cur_row = r[j];
                    if (prev_row != first_row) {    // If previous row seen is not first row,
                        y[r[j-1]] = v[j-1];         // then write partial sum to output
                    } else {
                        first_psum = v[j-1];        // Otherwise, store as first partial sum
                    }
                }
            }
            
            if (first_row == cur_row) {
                first_psum = v[blockDim.x - 1];     // Only one row was seen
            } else {
                last_psum = v[blockDim.x - 1];      // More than one row was seen
            }
            
            buf1[2*blockIdx.x] = first_psum;
            buf1[2*blockIdx.x + 1] = last_psum;
            row_buf1[2*blockIdx.x] = first_row;
            row_buf1[2*blockIdx.x + 1] = cur_row;
        }
        __syncthreads();
    
    }
*/
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
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
    cudaMemcpy(out_vec, d_out_vec, mat->M * sizeof(float), cudaMemcpyDeviceToHost);

    free(out_vec);
    cudaFree(d_rIndex);
    cudaFree(d_cIndex);
    cudaFree(d_val);
    cudaFree(d_out_vec);
}
