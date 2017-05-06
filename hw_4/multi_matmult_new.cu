#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_DIM 32
#define SCALING_FACTOR 10.0
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

double * creat_mat(int dimX, int dimY)
{
    int x;
    double * mat = (double *) malloc(dimX * dimY * sizeof(double));

    srand(time(NULL));

    for (x = 0; x < dimX * dimY; x++) {
        //mat[x] = float(rand()) / float(RAND_MAX) * SCALING_FACTOR;
        mat[x] = float(rand()) / float(RAND_MAX) * SCALING_FACTOR;
       //printf("%f\n", mat[x]);
    }
    return mat;
}

void if_mats_equal(double * A,  double * B, int rows, int cols)
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

void cpu_mat_mul(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols)
{
    double sum = 0.0;
    for (int i = 0; i < ARows; i++) {
        for (int j = 0; j < BCols; j++) {
            for (int k = 0; k < ACols; k++) {
                sum += A[i * ACols + k] * B[k * BCols + j];
                //C[i * BCols + j] += A[i * ACols + k] * B[k * BCols + j];
            }
            C[i * BCols + j] = double(int(sum) % MOD_BASE);
            sum = 0.0;
        }
    }
}

void print_mat(double * mat, int dimX, int dimY)
{
    for (int i = 0; i < dimX; i++) {
        for (int j = 0; j < dimY; j++) {
            printf("%2.2f ", mat[i * dimX + j]);
        }
        printf("\n");
    }
}

double * cpu_multi_mat_mult(int num_dim, int * dim_list, double ** mat_list) {

    int max_dim = find_max(dim_list, num_dim);
    double * output_mat1 = (double *) calloc(max_dim * max_dim, sizeof(double));
    double * output_mat2 = (double *) calloc(max_dim * max_dim, sizeof(double));

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







__device__
void MatMul(/* parameters */) {


}

/*
__global__ 
void matmult(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    if ((col < BCols) && (row < ARows)) {
        for (int i = 0; i < ACols; i++) {
            sum += A[row * ACols + i] * B[i * BCols + col];
        }
        C[row * BCols + col] = sum;
    }
}
*/
__global__
void matmult_general(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols)
{
    int num_elem_output = ARows * BCols;
    int C_elem_row = 0;
    int C_elem_col = 0;
    double sum = 0.0f;

    for (int n = threadIdx.x; n < num_elem_output; n+=NUM_THREADS) {
        C_elem_col = n % BCols;
        C_elem_row = (n + (BCols - C_elem_col)) / BCols - 1;
        
        for (int i = 0; i < ACols; i++) {
            sum += A[C_elem_row * ACols + i] * B[i * BCols + C_elem_col];
        }

        C[C_elem_row * ACols + C_elem_col] = sum;
        sum = 0.0f;
    }

}
/*
__global__
void gpu_seq_multi_matmult(int num_dim, int * dim_list, double ** mat_list, double * output_mat1, double * output_mat2)
{
    int grid_rows = (dim_list[0] + TILE_DIM - 1) / TILE_DIM;
    int grid_cols = (dim_list[2] + TILE_DIM - 1) / TILE_DIM;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    if (threadIdx.x == 0) {
    matmult<<<dimGrid, dimBlock>>>(mat_list[0], mat_list[1], output_mat, dim_list[0], dim_list[1], dim_list[1], dim_list[2]);
    cudaDeviceSynchronize();
    }
    __syncthreads();
    //cudaThreadSynchronize();
}
*/


int main()
{

    int num_dim = 100;
    int num_mat = num_dim - 1;
    int * mat_dim = def_mat_dim(num_dim);
    double ** mat_list = (double **) malloc((num_mat) * sizeof(double *));


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
    double * out_mat = (double *) malloc(mat_dim[0] * mat_dim[num_dim-1] * sizeof(double));
    double * d_out_mat;
    cudaMalloc((void **) &d_out_mat, mat_dim[0] * mat_dim[num_dim-1] * sizeof(double));

//    printf("Allocating space for each matrix, and storing pointer address of matrices on the host\n");
    double ** int_mat_list = (double **) malloc(num_mat * sizeof(double *));
    for (k = 0; k < num_mat; k++) {
        cudaMalloc((void **)&int_mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(double));
        cudaMemcpy(int_mat_list[k], mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(double), cudaMemcpyHostToDevice);
    }

//    printf("Copying pointer addresses of matrices from host to device\n");
    double ** d_mat_list;
    cudaMalloc(&d_mat_list, num_mat * sizeof(double *));
    cudaMemcpy(d_mat_list, int_mat_list, num_mat * sizeof(double *), cudaMemcpyHostToDevice);

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
        cudaMalloc((void **)&d_mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(double));
        //cudaMemcpy(d_mat_list[k], mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(double), cudaMemcpyHostToDevice);

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

    double * cpu_mat = cpu_multi_mat_mult(num_dim, mat_dim, mat_list);
    printf("%d %d\n", mat_dim[0], mat_dim[num_dim-1]);
    print_mat(cpu_mat, mat_dim[0], mat_dim[num_dim-1]);
    printf("\n");
/*
    printf("%d %d %d\n", mat_dim[0], mat_dim[1], mat_dim[2]);
    //matmult<<<dimGrid, dimBlock>>>(int_mat_list[0], int_mat_list[1], d_out_mat, mat_dim[0], mat_dim[1], mat_dim[1], mat_dim[2]);
    matmult_general<<<1, NUM_THREADS>>>(int_mat_list[0], int_mat_list[1], d_out_mat, mat_dim[0], mat_dim[1], mat_dim[1], mat_dim[2]);
    cudaThreadSynchronize();
    //multi_matmult<<<1, NUM_THREADS>>>(num_dim, d_mat_dim, d_mat_list, d_out_mat);
    //gpuErrchk(cudaPeekAtLastError());

    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
       // print the CUDA error message and exit
       printf("CUDA error: %s\n", cudaGetErrorString(error));
       exit(-1);
    }

    cudaMemcpy(out_mat, d_out_mat, mat_dim[0] * mat_dim[num_dim-1] * sizeof(double), cudaMemcpyDeviceToHost);
    print_mat(out_mat, mat_dim[0], mat_dim[num_dim-1]);
    printf("\n");

    if_mats_equal(out_mat, cpu_mat, mat_dim[0], mat_dim[2]);
*/    
    return 0;

}
