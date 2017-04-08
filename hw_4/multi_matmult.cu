#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_DIM 1024
#define SCALING_FACTOR 100.0


int * def_mat_dim(int k)
{
    int * dim = (int *) malloc(k * sizeof(int));
    int i;
    srand(time(NULL));

    for (i = 0; i < k; i++)
    {
        dim[i] = rand() % MAX_DIM + 1;
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



int main()
{

    int num_dim = 2;
    int * mat_dim = def_mat_dim(num_dim);
    double ** mat_list = (double **) malloc((num_dim-1) * sizeof(double *));

    int k;
    for (k = 0; k < num_dim-1; k++) {
        //printf("================= MATRIX %d ====================\n", k);
        mat_list[k] = creat_mat(mat_dim[k], mat_dim[k+1]);
    }

    double * out_mat;

    int * d_mat_dim;
    cudaMalloc(&d_mat_dim, num_dim * sizeof(int));
    cudaMemcpy(d_mat_dim, mat_dim, num_dim * sizeof(int), cudaMemcpyHostToDevice);

    double ** d_mat_list;
    cudaMalloc(&d_mat_list, num_dim-1 * sizeof(double *));

    for (k = 0; k < num_dim - 1; k++) {
        cudaMalloc(&d_mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(double));
        cudaMemcpy(d_mat_list[k], mat_list[k], num_dim-1 * sizeof(double), cudaMemcpyHostToDevice);
    }






    return 0;

}
