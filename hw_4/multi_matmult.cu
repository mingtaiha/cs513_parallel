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

double ** creat_mat(int dimX, int dimY)
{
    int x, y;
    double ** mat = (double **) malloc(dimX * sizeof(double *));

    int i;
    for (i = 0; i < dimX; i++) {
        mat[i] = (double *) malloc(dimY * sizeof(double));
    }

    srand(time(NULL));

    for (x = 0; x < dimX; x++) {
        for (y = 0; y < dimY; y++) {
            mat[x][y] = float(rand()) / float(RAND_MAX) * SCALING_FACTOR;
            //printf("%f\n", mat[x][y]);
        }
    }
    return mat;
}

__device__
void matmult(/* parameters */) {


}


__global__
void multi_matmult(/* parameters */)
{


}



int main()
{

    int num_dim = 10;
    int * mat_dim = def_mat_dim(num_dim);
    double *** mat_list = (double ***) malloc((num_dim-1) * sizeof(double **));

    int k;
    for (k = 0; k < num_dim-1; k++) {
        printf("================= MATRIX %d ====================\n", k);
        mat_list[k] = creat_mat(mat_dim[k], mat_dim[k+1]);
    }







    return 0;

}
