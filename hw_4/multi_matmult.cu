#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_DIM 32
#define SCALING_FACTOR 100.0
#define TILE_DIM 32
#define NUM_THREADS 1024

int * def_mat_dim(int k)
{
    int * dim = (int *) malloc(k * sizeof(int));
    int i;
    srand(time(NULL));

    for (i = 0; i < k; i++)
    {
        dim[i] = (rand() % MAX_DIM) + 1;
        printf("%d\n", dim[i]);
    }
    return dim;
}

double * creat_mat(int dimX, int dimY)
{
    int x;
    double * mat = (double *) malloc(dimX * dimY * sizeof(double));

    srand(time(NULL));

    for (x = 0; x < dimX * dimY; x++) {
        mat[x] = float(rand()) / float(RAND_MAX) * SCALING_FACTOR;
       //printf("%f\n", mat[x]);
    }
    return mat;
}

__device__
void matmult(/* parameters */) {


}


__global__
void multi_matmult(int num_dim, int * dim_list, double ** mat_list)
{

}


__global__ void MatMul(double* A, double* B, double* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ double As[TILE_DIM][TILE_DIM];
    __shared__ double Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;



}



int main()
{

    int num_dim = 3;
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
        printf("%d %d\n", mat_dim[k], mat_dim[k+1]);
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
    printf("%d %d %d\n", mat_dim[0], mat_dim[1], mat_dim[2]);
    MatMul<<<2, NUM_THREADS>>>(int_mat_list[0], int_mat_list[1], d_out_mat, mat_dim[0], mat_dim[1], mat_dim[1], mat_dim[2], mat_dim[0], mat_dim[2]);
    cudaThreadSynchronize();


    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
       // print the CUDA error message and exit
       printf("CUDA error: %s\n", cudaGetErrorString(error));
       exit(-1);
    }

    return 0;

}
