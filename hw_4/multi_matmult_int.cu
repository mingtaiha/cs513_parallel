#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_DIM 32
#define SCALING_FACTOR 256
#define TILE_DIM 32
#define NUM_THREADS 1024
#define MOD_BASE 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int find_max(int * arr, int num_elem)
{
    int max = 0;
    for (int i = 0; i < num_elem; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

int * def_mat_dim(int k)
{
    int * dim = (int *) malloc(k * sizeof(int));
    int i;
    //srand(time(NULL));

    for (i = 0; i < k; i++)
    {
        //dim[i] = 10;
        dim[i] = (rand() % MAX_DIM) + 1;
        //printf("%d\n", dim[i]);
    }
    return dim;
}

int * creat_mat(int dimX, int dimY)
{
    int x;
    int * mat = (int *) malloc(dimX * dimY * sizeof(int));

    srand(time(NULL));

    for (x = 0; x < dimX * dimY; x++) {
        mat[x] = rand() % MOD_BASE;
        //mat[x] = (rand() % MAX_DIM) * SCALING_FACTOR;
        //printf("%d ", mat[x]);
    }
    return mat;
}

void if_mats_equal(int * A,  int * B, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (A[i * rows + j] != B[i * rows + j]) { 
                printf("Matrices are not equal\n"); 
                return;
            }
        }
    }
    printf("Matrices are equal\n");
}

void cpu_mat_mul(int* A, int* B, int* C, int ARows, int ACols, int BRows, int BCols)
{
    int sum = 0;
    for (int i = 0; i < ARows; i++) {
        for (int j = 0; j < BCols; j++) {
            for (int k = 0; k < ACols; k++) {
                //prod = ((A[i * ACols + k] % MOD_BASE) * (B[k * BCols + j] % MOD_BASE)) % MOD_BASE;
                sum += (A[i * ACols + k] * B[k * BCols + j]);
            }
            //C[i * BCols + j] = sum % MOD_BASE;
            C[i * BCols + j] = sum;
            sum = 0;
        }
    }
}

void print_mat(int * mat, int dimX, int dimY)
{
    for (int i = 0; i < dimX; i++) {
        for (int j = 0; j < dimY; j++) {
            printf("%d  ", mat[i * dimX + j]);
        }
        printf("\n");
    }
}

int * cpu_multi_mat_mult(int num_dim, int * dim_list, int ** mat_list) {

    int max_dim = find_max(dim_list, num_dim);
    int * output_mat1 = (int *) calloc(max_dim * max_dim, sizeof(int));
    int * output_mat2 = (int *) calloc(max_dim * max_dim, sizeof(int));

    cpu_mat_mul(mat_list[0], mat_list[1], output_mat1, dim_list[0], dim_list[1], dim_list[1], dim_list[2]);
    int num_rows = dim_list[0];
    int num_cols = dim_list[2];

    //print_mat(output_mat1, num_rows, num_cols);

    int num_mult;
    for (num_mult = 1; num_mult < num_dim - 2; num_mult++) {
        if (num_mult % 2 == 1) {
            cpu_mat_mul(output_mat1, mat_list[num_mult + 1], output_mat2, num_rows, num_cols, dim_list[num_mult + 1] , dim_list[num_mult + 2]);
        }
        else {
            cpu_mat_mul(output_mat2, mat_list[num_mult + 1], output_mat1, num_rows, num_cols, dim_list[num_mult + 1] , dim_list[num_mult + 2]);
        }
        num_cols = dim_list[num_mult + 2];
    }

    //printf("%d %d\n", num_rows, num_cols);
    if (num_mult % 2 == 1) {
        free(output_mat2);
        return output_mat1;
    }
    else {
        free(output_mat1);
        return output_mat2;
    }
}

/*
__global__ 
void matmult(int* A, int* B, int* C, int ARows, int ACols, int BRows, int BCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0.0;

    if ((col < BCols) && (row < ARows)) {
        for (int i = 0; i < ACols; i++) {
            sum += A[row * ACols + i] * B[i * BCols + col];
        }
        C[row * BCols + col] = sum;
    }
}
*/
__device__
void matmult(int* A, int* B, int* C, int ARows, int ACols, int BRows, int BCols)
{
    int num_elem_output = ARows * BCols;
    int C_elem_row = 0;
    int C_elem_col = 0;
    int sum = 0;

    for (int n = threadIdx.x; n < num_elem_output; n+=NUM_THREADS) {
        C_elem_col = n % BCols;
        C_elem_row = (n + (BCols - C_elem_col)) / BCols - 1;
        
        for (int i = 0; i < ACols; i++) {
            //sum += ((A[C_elem_row * ACols + i] % MOD_BASE) * (B[i * BCols + C_elem_col] % MOD_BASE)) % MOD_BASE;
            //sum += (A[C_elem_row * ACols + i] * B[i * BCols + C_elem_col]) % MOD_BASE;
            sum += A[C_elem_row * ACols + i] * B[i * BCols + C_elem_col];
        }

        C[C_elem_row * BCols + C_elem_col] = sum;
        sum = 0;
    }
    __syncthreads();
}

__global__
void gpu_seq_multi_matmult(int num_dim, int * dim_list, int ** mat_list, int * output_mat1, int * output_mat2)
{

    matmult(mat_list[0], mat_list[1], output_mat1, dim_list[0], dim_list[1], dim_list[1], dim_list[2]);
    __syncthreads();
    //cudaThreadSynchronize();
}


int main()
{

    int num_dim = 3;
    int num_mat = num_dim - 1;
    int * mat_dim = def_mat_dim(num_dim);
    int ** mat_list = (int **) malloc((num_mat) * sizeof(int *));
    int max_dim = find_max(mat_dim, num_dim);

//    printf("Copying matrix dimensions to device\n");

    int * d_mat_dim;
    cudaMalloc((void **)&d_mat_dim, num_dim * sizeof(int));
    cudaMemcpy(d_mat_dim, mat_dim, num_dim * sizeof(int), cudaMemcpyHostToDevice);

//    printf("Creating Matrix from on host\n");

    int k;
    for (k = 0; k < num_mat; k++) {
        //printf("================= MATRIX %d ====================\n", k);
        //printf("%d %d\n", mat_dim[k], mat_dim[k+1]);
        mat_list[k] = creat_mat(mat_dim[k], mat_dim[k+1]);
    }
    
//    printf("Allocating space to store output matrix\n");
    int * out_mat = (int *) calloc(max_dim * max_dim, sizeof(int));
    int * d_out_mat1, * d_out_mat2;
    cudaMalloc((void **) &d_out_mat1, max_dim * max_dim * sizeof(int));
    cudaMalloc((void **) &d_out_mat2, max_dim * max_dim * sizeof(int));
    cudaMemcpy(d_out_mat1, out_mat, max_dim * max_dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_mat2, out_mat, max_dim * max_dim * sizeof(int), cudaMemcpyHostToDevice);

//    printf("Allocating space for each matrix, and storing pointer address of matrices on the host\n");
    int ** int_mat_list = (int **) malloc(num_mat * sizeof(int *));
    for (k = 0; k < num_mat; k++) {
        cudaMalloc((void **)&int_mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(int));
        cudaMemcpy(int_mat_list[k], mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(int), cudaMemcpyHostToDevice);
    }

//    printf("Copying pointer addresses of matrices from host to device\n");
    int ** d_mat_list;
    cudaMalloc(&d_mat_list, num_mat * sizeof(int *));
    cudaMemcpy(d_mat_list, int_mat_list, num_mat * sizeof(int *), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
       // print the CUDA error message and exit
       printf("CUDA error: %s\n", cudaGetErrorString(error));
       exit(-1);
    }

/*
    for (k = 0; k < num_dim-1; k++) {
        printf("%d %d %d %d\n", k, mat_dim[k], mat_dim[k+1], &d_mat_list[k]);
        cudaMalloc((void **)&d_mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(int));
        //cudaMemcpy(d_mat_list[k], mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(int), cudaMemcpyHostToDevice);

        if(error != cudaSuccess)
        {
           // print the CUDA error message and exit
           printf("CUDA error: %s\n", cudaGetErrorString(error));
           exit(-1);
        }
    }
    printf("After d_mat_list\n");
*/

//    printf("At the kernel call\n");
/*
    int grid_rows = (mat_dim[0] + TILE_DIM - 1) / TILE_DIM;
    int grid_cols = (mat_dim[2] + TILE_DIM - 1) / TILE_DIM;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(TILE_DIM, TILE_DIM);
*/

    int * cpu_mat = cpu_multi_mat_mult(num_dim, mat_dim, mat_list);
    printf("%d %d\n", mat_dim[0], mat_dim[num_dim-1]);
    print_mat(cpu_mat, mat_dim[0], mat_dim[num_dim-1]);
    printf("\n");

    //printf("%d %d %d\n", mat_dim[0], mat_dim[1], mat_dim[2]);
    //matmult<<<dimGrid, dimBlock>>>(int_mat_list[0], int_mat_list[1], d_out_mat, mat_dim[0], mat_dim[1], mat_dim[1], mat_dim[2]);
    //matmult<<<1, NUM_THREADS>>>(int_mat_list[0], int_mat_list[1], d_out_mat, mat_dim[0], mat_dim[1], mat_dim[1], mat_dim[2]);
    //cudaThreadSynchronize();
    //multi_matmult<<<1, NUM_THREADS>>>(num_dim, d_mat_dim, d_mat_list, d_out_mat);
    //gpuErrchk(cudaPeekAtLastError());

    gpu_seq_multi_matmult<<<1, NUM_THREADS>>>(num_dim, d_mat_dim, d_mat_list, d_out_mat1, d_out_mat2);
    cudaThreadSynchronize();

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
       // print the CUDA error message and exit
       printf("CUDA error: %s\n", cudaGetErrorString(error));
       exit(-1);
    }

    cudaMemcpy(out_mat, d_out_mat1, mat_dim[0] * mat_dim[num_dim-1] * sizeof(int), cudaMemcpyDeviceToHost);
    print_mat(out_mat, mat_dim[0], mat_dim[num_dim-1]);
    printf("\n");

    if_mats_equal(out_mat, cpu_mat, mat_dim[0], mat_dim[2]);
    
    return 0;

}
