#include "genresult.cuh"
#include <sys/time.h>

#define BLOCK_SIZE 1024
#define min(a, b) ((a) < (b)) ? (a) : (b)
#define max(a, b) ((a) > (b)) ? (a) : (b)

typedef struct A_cont {
    int r;
    int c;
    float v;
} A_cont;



__device__ void filter_sums_old(const int * rows, const float * vals, float * y, int * r_buf, float * v_buf) {

    
    // Going through each thread and filtering out row indices where all products
    // in the row has been summed, and thus does not need to be considered further
    // This step has two purposes: 1) it filters output elements which do not need
    // to be reduced, 2) it guarantees that at most two elements will be written
    // to the row buffer and computation buffer. The second step makes the
    // aggregation of partial sums much more predictable
    if (threadIdx.x == 0) {
        int first_row = rows[0];                // First row encountered in shared memory
        int prev_row = rows[0];                 // Previous row encountered when scanning shared memory
        int cur_row = rows[0];                  // Newest row encountered when scanning shared memory    
        int first_psum = 0;                     // First partial sum
        int last_psum = 0;                      // Last partial sum
        for (int j = 1; j < blockDim.x; j++) {
            if (rows[j] != rows[j-1]) {             // If rows change
                prev_row = cur_row;                 // Update previous and current row
                cur_row = rows[j];
                if (prev_row != first_row) {        // If previous row seen is not first row,
                    y[rows[j-1]] = vals[j-1];       // then write partial sum to output
                } else {
                    first_psum = vals[j-1];         // Otherwise, store as first partial sum
                }
            }
        }
        
        if (first_row == cur_row) {
            first_psum = vals[blockDim.x - 1];      // Only one row was seen
        } else {
            last_psum = vals[blockDim.x - 1];       // More than one row was seen
        }
        
        v_buf[2*blockIdx.x] = first_psum;           // Write to intermediate buffer
        v_buf[2*blockIdx.x + 1] = last_psum;
        r_buf[2*blockIdx.x] = first_row;
        r_buf[2*blockIdx.x + 1] = cur_row;
    }
}





__device__ void segmented_scan(const int lane, const int nnz, const int thr_id, const int * rows, float * vals) {

    if (thr_id < nnz) {
        for (int i = 1; i < blockDim.x; i *= 2) {
            if ((lane >= i) && (rows[threadIdx.x] == rows[threadIdx.x - i])) {
                vals[threadIdx.x] += vals[threadIdx.x - i];
            }
        }
    }
}


__device__ void filter_sums(const int nnz, const int thr_id, const int bid, const int * rows, const float * vals, float * y, int * r_buf, float * v_buf) {

    // Going through each thread and filtering out row indices where all products
    // in the row has been summed, and thus does not need to be considered further
    // This step has two purposes: 1) it filters output elements which do not need
    // to be reduced, 2) it guarantees that at most two elements will be written
    // to the row buffer and computation buffer. The second step makes the
    // aggregation of partial sums much more predictable. Specifically, we only
    // need to keep track at most two rows, the first row and the last row which
    // appears

    if (thr_id < nnz) {
        int first_row = rows[0];
        int last_row = rows[blockDim.x - 1];
        int cur_row = rows[threadIdx.x];

        if (threadIdx.x != blockDim.x - 1) {    // Every element other than the last element looks right
            int next_row = rows[threadIdx.x + 1];   // Check row of associated right thread
            if (cur_row != next_row) {              // Indicate current thread is last element in row
                if (cur_row == first_row) {         // Partial sum of first row stored in shared memory
                    v_buf[2*bid] = vals[threadIdx.x];    // Write to buffer
                    r_buf[2*bid] = vals[threadIdx.x];
                } else {
                    y[rows[threadIdx.x]] = vals[threadIdx.x];   // Write to output
                }
            }
        } else {        // Last element is the partial sum of last row in shared memory
            r_buf[2*bid + 1] = rows[threadIdx.x];        // Update row
            if (first_row != last_row) {
                v_buf[2*bid + 1] = vals[threadIdx.x];    // First row mentioned is not last row mentioned
            } else {
                v_buf[2*bid + 1] = 0;                    // Marked as 0 in order to allow segmented scan to work
            } 
        }
    }
}


__global__ void segmented_spmv_kernel(const int nnz, const int * rows, const int * cols, const float * A, const float * x, float * y, int * row_buf0, float * val_buf0, int * row_buf1, float * val_buf1){
    /*Put your kernel(s) implementation here, you don't have to use exactly the
 * same kernel name */

    __shared__ float v[BLOCK_SIZE];
    __shared__ int r[BLOCK_SIZE];

    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int thread_num = blockDim.x * gridDim.x;
    int iter = nnz % thread_num ? nnz/thread_num + 1 : nnz/thread_num;
  
    // Iterating through all N elements, using all threads T, where T can be less than N
    for (int i = 0; i < iter; i++)
    {
        // Storing products and row index in shared memory
        int prodid = thread_id + i * thread_num;
        int blockid = blockIdx.x + i * gridDim.x;
        if (prodid < nnz) {
            v[threadIdx.x] = A[prodid] * x[cols[prodid]];
            r[threadIdx.x] = rows[prodid];
        
        __syncthreads();
        
        // First pass of segmented partial sum
        segmented_scan(thread_id % blockDim.x, nnz, prodid, r, v);
        __syncthreads();
        
        // Writing partial sums whose partition is entirely in the block to output
        // and storing sums who needs to be reduced
        filter_sums(nnz, prodid, blockid, r, v, y, row_buf0, val_buf0);
        __syncthreads();
        
        }
    }
/*
    int num_blocks = nnz % blockDim.x ? nnz/blockDim.x + 1 : nnz/blockDim.x;
    int red_nnz = num_blocks * 2;
    int red_iter = red_nnz % thread_num ? red_nnz/thread_num + 1: red_nnz/thread_num;
    int buf_to_use = 0;

    while (num_blocks > 1) {
        for (int j = 0; j < red_iter; j++) 
        {
            if (buf_to_use == 0) {
                // Taking elements from buffer and storing them in shared memory
                int red_prodid = thread_id + j * thread_num;
                if (red_prodid < red_nnz) {
                    v[threadIdx.x] = val_buf0[red_prodid];
                    r[threadIdx.x] = row_buf0[red_prodid];
                
                __syncthreads();

                // Performing segmented partial sum again
                segmented_scan(thread_id % blockDim.x, red_nnz, red_prodid, r, v);
                __syncthreads();

                // Writing partial sums whose partition is entirely in the block to output
                // and storing sums who need to be reduced
                filter_sums(red_nnz, red_prodid, r, v, y, row_buf1, val_buf1);
                __syncthreads();
                }
            } else {
                // Taking elements from buffer and storing them in shared memory
                int red_prodid = thread_id + j * thread_num;
                if (red_prodid < red_nnz) {
                    v[threadIdx.x] = val_buf1[red_prodid];
                    r[threadIdx.x] = row_buf1[red_prodid];
                
                __syncthreads();

                // Performing segmented partial sum again
                segmented_scan(thread_id % blockDim.x, red_nnz, red_prodid, r, v);
                __syncthreads();

                // Writing partial sums whose partition is entirely in the block to output
                // and storing sums who need to be reduced
                filter_sums(red_nnz, red_prodid, r, v, y, row_buf0, val_buf0);
                __syncthreads();
                }
            }
        }
        num_blocks = red_nnz % blockDim.x ? red_nnz/blockDim.x + 1 : red_nnz/blockDim.x;
        red_nnz = num_blocks * 2;
        red_iter = red_nnz % thread_num ? red_nnz/thread_num + 1: red_nnz/thread_num;
        buf_to_use = (buf_to_use + 1) % 2;
    }

    if (iter == 1) {

    }
*/


}

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

/*
    for (int k = 0; k < mat->nz; k++) {
        printf("row: %d\t col: %d\t, val: %f\n", mat->rIndex[k], mat->cIndex[k], mat->val[k]);
    }
*/
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

    // Allocating buffer space when reducing rIndex and val vectors
    int * d_row_buf0, * d_row_buf1;
    float * d_val_buf0, * d_val_buf1;
    int buf_size = (mat->nz / blockSize + 1)* 2;
    int * int_zero_vec = (int *) calloc(buf_size, sizeof(int));
    float * flt_zero_vec = (float *) calloc(buf_size, sizeof(float));
    cudaMalloc(&d_row_buf0, buf_size * sizeof(int));
    cudaMalloc(&d_row_buf1, buf_size * sizeof(int));
    cudaMemcpy(d_row_buf0, int_zero_vec, buf_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_val_buf0, buf_size * sizeof(float));
    cudaMalloc(&d_val_buf1, buf_size * sizeof(float));
    cudaMemcpy(d_val_buf0, flt_zero_vec, buf_size * sizeof(float), cudaMemcpyHostToDevice);



	/* Sample timing code */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernel(s)*/

    segmented_spmv_kernel<<<blockNum, blockSize>>>(mat->nz, d_rIndex, d_cIndex, d_val, d_vec, d_out_vec, d_row_buf0, d_val_buf0, d_row_buf1, d_val_buf1);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
          printf("CUDA error: %s\n", cudaGetErrorString(error));
          exit(-1);
    }


    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    int * out_row_buf = (int *) malloc (buf_size * sizeof(int));
    float * out_val_buf = (float *) malloc (buf_size * sizeof(float));

    cudaMemcpy(out_row_buf, d_row_buf0, buf_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_val_buf, d_val_buf0, buf_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < buf_size; i++) {
        printf("%d\t%d\n", i, out_row_buf[i]);
    }
    printf("newnewnew\n");
    for (int i = 0; i < buf_size; i++) {
        printf("%d\t%f\n", i, out_val_buf[i]);
    }


    /*Deallocate, please*/
    cudaMemcpy(res->val, d_out_vec, mat->M * sizeof(float), cudaMemcpyDeviceToHost);

    free(out_vec);
    cudaFree(d_rIndex);
    cudaFree(d_cIndex);
    cudaFree(d_val);
    cudaFree(d_out_vec);
}
