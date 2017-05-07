#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_DIM 20
#define SCALING_FACTOR 256
#define NUM_THREADS 1024
#define MOD_BASE 10007

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
    srand(time(NULL));

    for (i = 0; i < k; i++)
    {
        dim[i] = (rand() % MAX_DIM) + 1;
        //printf("%d\n", dim[i]);
    }
    return dim;
}

int * equipartition(int k, int blocks) 
{
    float div = float(k) / float(blocks);
    int div_int = int(div);
    float rem = div - float(div_int);

    int * partition = (int *) malloc((blocks + 1) * sizeof(int));
    srand(time(NULL));

    partition[0] = 0;
    partition[blocks] = k-1;

    int cur_index = 0;
    float round_factor = 0.0;
    for (int i = 1; i < blocks; i++) {
        cur_index += div;
        round_factor = float(rand()) / float(RAND_MAX);
        if (round_factor < rem) {
            cur_index += 1;
        }
        partition[i] = cur_index;
    }
    return partition;
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
            if (A[i * cols + j] != B[i * cols + j]) { 
                printf("Matrices are not equal\n"); 
                //printf("%d %d\n", i, j);
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
                sum += (A[i * ACols + k] * B[k * BCols + j]) % MOD_BASE;
                //sum += A[i * ACols + k] * B[k * BCols + j];
            }
            //printf("%d %d\n", i, j);
            C[i * BCols + j] = sum % MOD_BASE;
            //C[i * BCols + j] = sum;
            sum = 0;
        }
    }
}

void print_mat(int * mat, int dimX, int dimY)
{
    for (int i = 0; i < dimX; i++) {
        for (int j = 0; j < dimY; j++) {
            printf("%d  ", mat[i * dimY + j]);
        }
        printf("\n");
    }
}

int * cpu_multi_mat_mult(int num_dim, int * dim_list, int ** mat_list, int start = 0) {

    int max_dim = find_max(dim_list, num_dim);
    int * output_mat1 = (int *) calloc(max_dim * max_dim, sizeof(int));
    int * output_mat2 = (int *) calloc(max_dim * max_dim, sizeof(int));

    cpu_mat_mul(mat_list[start], mat_list[start + 1], output_mat1, dim_list[start], dim_list[start + 1], dim_list[start + 1], dim_list[start + 2]);
    int num_rows = dim_list[start];
    int num_cols = dim_list[start + 2];

    //print_mat(output_mat1, num_rows, num_cols);

    int num_mult;
    for (num_mult = 1; num_mult < num_dim - 2; num_mult++) {
        //printf("multiplied %d matrices\n", num_mult + 1);
        if (num_mult % 2 == 1) {
            cpu_mat_mul(output_mat1, mat_list[start + num_mult + 1], output_mat2, num_rows, num_cols, dim_list[start + num_mult + 1] , dim_list[start + num_mult + 2]);
        }
        else {
            cpu_mat_mul(output_mat2, mat_list[start + num_mult + 1], output_mat1, num_rows, num_cols, dim_list[start + num_mult + 1] , dim_list[start + num_mult + 2]);
        }
        num_cols = dim_list[start + num_mult + 2];
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
            sum += (A[C_elem_row * ACols + i] * B[i * BCols + C_elem_col]) % MOD_BASE;
            //sum += A[C_elem_row * ACols + i] * B[i * BCols + C_elem_col];
        }
        C[C_elem_row * BCols + C_elem_col] = sum % MOD_BASE;
        //C[C_elem_row * BCols + C_elem_col] = sum;
        sum = 0;
        __syncthreads();
    }
    __syncthreads();
}

__global__
void gpu_seq_multi_matmult(int num_dim, int * dim_list, int ** mat_list, int * output_mat1, int * output_mat2)
{

    matmult(mat_list[0], mat_list[1], output_mat1, dim_list[0], dim_list[1], dim_list[1], dim_list[2]);
    __syncthreads();

    int num_mult;
    int num_rows = dim_list[0];
    int num_cols = dim_list[2];
    for (num_mult = 1; num_mult < num_dim - 2; num_mult++) {
        if (num_mult % 2 == 1) {
            matmult(output_mat1, mat_list[num_mult + 1], output_mat2, num_rows, num_cols, dim_list[num_mult + 1], dim_list[num_mult + 2]);
        } else {
            matmult(output_mat2, mat_list[num_mult + 1], output_mat1, num_rows, num_cols, dim_list[num_mult + 1], dim_list[num_mult + 2]);
        }
        num_cols = dim_list[num_mult + 2];
        __syncthreads();
    }
}

__global__
void gpu_par_multi_matmult(int start_dim_idx, int end_dim_idx, int * dim_list, int ** mat_list, int * output_mat1, int * output_mat2) //, int* i)
{

    matmult(mat_list[start_dim_idx], mat_list[start_dim_idx + 1], output_mat1, dim_list[start_dim_idx], dim_list[start_dim_idx + 1], dim_list[start_dim_idx + 1], dim_list[start_dim_idx + 2]);
/*
    if (threadIdx.x == 0) { 
        i[0]++; 
        i[1] = start_dim_idx;
        i[2] = end_dim_idx;
        }
*/
    __syncthreads();


    int num_mult;
    int num_rows = dim_list[start_dim_idx];
    int num_cols = dim_list[start_dim_idx + 2];
    int count = 1;
//    for (num_mult = start_dim_idx + 1; count < end_dim_idx - start_dim_idx - 2; num_mult++) {
    for (int count = 1; count < (end_dim_idx - start_dim_idx - 2); count++) {
        if (count % 2 == 1) {
            matmult(output_mat1, mat_list[start_dim_idx + count + 1], output_mat2, num_rows, num_cols, dim_list[start_dim_idx + count + 1], dim_list[start_dim_idx + count + 2]);
        } else {
            matmult(output_mat2, mat_list[start_dim_idx + count + 1], output_mat1, num_rows, num_cols, dim_list[start_dim_idx + count + 1], dim_list[start_dim_idx + count + 2]);
            //matmult(output_mat2, mat_list[num_mult + 1], output_mat1, num_rows, num_cols, dim_list[num_mult + 1], dim_list[num_mult + 2]);
        }
        num_cols = dim_list[start_dim_idx + count + 2];
/*        
        if (threadIdx.x == 0) { 
            i[0]++; 
            i[1] = num_rows;
            i[2] = num_cols;
            }
*/
        __syncthreads();
    }

}


int main()
{

    int nblocks = 1;
    int num_dim = 45;
    int num_mat = num_dim - 1;
    int * mat_dim = def_mat_dim(num_dim);
    int ** mat_list = (int **) malloc((num_mat) * sizeof(int *));
    int max_dim = find_max(mat_dim, num_dim);

    printf("Copying matrix dimensions to device\n");

    int * d_mat_dim;
    cudaMalloc((void **)&d_mat_dim, num_dim * sizeof(int));
    cudaMemcpy(d_mat_dim, mat_dim, num_dim * sizeof(int), cudaMemcpyHostToDevice);

    printf("Creating Matrix from on host\n");

    int k;
    for (k = 0; k < num_mat; k++) {
        //printf("================= MATRIX %d ====================\n", k);
        //printf("%d %d\n", mat_dim[k], mat_dim[k+1]);
        mat_list[k] = creat_mat(mat_dim[k], mat_dim[k+1]);
        //printf("%d %d\n", mat_dim[k], mat_dim[k+1]);
        //print_mat(mat_list[k], mat_dim[k], mat_dim[k+1]);
    }
    
    printf("Allocating space to store output matrix\n");
    int * out_mat = (int *) calloc(max_dim * max_dim, sizeof(int));
    int * d_out_mat1, * d_out_mat2;
    cudaMalloc((void **) &d_out_mat1, max_dim * max_dim * sizeof(int));
    cudaMalloc((void **) &d_out_mat2, max_dim * max_dim * sizeof(int));
    cudaMemcpy(d_out_mat1, out_mat, max_dim * max_dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_mat2, out_mat, max_dim * max_dim * sizeof(int), cudaMemcpyHostToDevice);

    printf("Allocating space for each matrix, and storing pointer address of matrices on the host\n");
    int ** int_mat_list = (int **) malloc(num_mat * sizeof(int *));
    for (k = 0; k < num_mat; k++) {
        cudaMalloc((void **)&int_mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(int));
        cudaMemcpy(int_mat_list[k], mat_list[k], mat_dim[k] * mat_dim[k+1] * sizeof(int), cudaMemcpyHostToDevice);
    }

    printf("Copying pointer addresses of matrices from host to device\n");
    int ** d_mat_list;
    cudaMalloc(&d_mat_list, num_mat * sizeof(int *));
    cudaMemcpy(d_mat_list, int_mat_list, num_mat * sizeof(int *), cudaMemcpyHostToDevice);


    printf("Allocating a set of intermediate arrays\n");
//  Allocating a set of intermediate 
    int ** int_mat1, ** int_mat2;
    int_mat1 = (int **) malloc(nblocks * sizeof(int *));
    int_mat2 = (int **) malloc(nblocks * sizeof(int *));
    for (k = 0; k < nblocks; k++) {
        cudaMalloc((void **)&int_mat1[k], max_dim * max_dim * sizeof(int));
        cudaMalloc((void **)&int_mat2[k], max_dim * max_dim * sizeof(int));
        cudaMemcpy(int_mat1[k], out_mat, max_dim * max_dim * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(int_mat2[k], out_mat, max_dim * max_dim * sizeof(int), cudaMemcpyHostToDevice);
    }

    printf("allocating final collection of intermediate arrays\n");
    int ** int_mat_final = (int **) malloc(nblocks * sizeof(int *));
    int ** d_int_mat_final;
    cudaMalloc((void **)&d_int_mat_final, nblocks * sizeof(int *));


    int * mat_list_partition = equipartition(num_dim, nblocks);
    printf("partition\n");
    print_mat(mat_list_partition, 1, nblocks + 1);

    printf("mat_dim\n");
    print_mat(mat_dim, 1, num_dim);
    int * int_mat_dim = (int *) malloc((nblocks + 1) * sizeof(int));
    for (int i = 0; i < nblocks + 1; i++) {
        int_mat_dim[i] = mat_dim[mat_list_partition[i]];
    }
    printf("partition mat_dim\n");
    print_mat(int_mat_dim, 1, nblocks + 1);
    printf("\n");
    int * d_int_mat_dim;
    cudaMalloc((void **)&d_int_mat_dim, (nblocks + 1) * sizeof(int));
    cudaMemcpy(d_int_mat_dim, int_mat_dim, (nblocks + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int * d_int_output_mat1, * d_int_output_mat2;
    cudaMalloc((void **)&d_int_output_mat1, max_dim * max_dim * sizeof(int));
    cudaMalloc((void **)&d_int_output_mat2, max_dim * max_dim * sizeof(int));
    cudaMemcpy(d_int_output_mat1, out_mat, max_dim * max_dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_int_output_mat2, out_mat, max_dim * max_dim * sizeof(int), cudaMemcpyHostToDevice);



    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
       // print the CUDA error message and exit
       printf("CUDA error: %s\n", cudaGetErrorString(error));
       exit(-1);
    }

    int * cpu_mat = cpu_multi_mat_mult(num_dim, mat_dim, mat_list);
    //printf("%d %d\n", mat_dim[0], mat_dim[num_dim-1]);
    printf("%d %d\n", mat_dim[0], mat_dim[num_dim-1]);
    printf("cpu seq\n");
    print_mat(cpu_mat, mat_dim[0], mat_dim[num_dim-1]);
    printf("\n");

    //printf("%d %d %d\n", mat_dim[0], mat_dim[1], mat_dim[2]);
    //matmult<<<dimGrid, dimBlock>>>(int_mat_list[0], int_mat_list[1], d_out_mat, mat_dim[0], mat_dim[1], mat_dim[1], mat_dim[2]);
    //matmult<<<1, NUM_THREADS>>>(int_mat_list[0], int_mat_list[1], d_out_mat, mat_dim[0], mat_dim[1], mat_dim[1], mat_dim[2]);
    //cudaThreadSynchronize();
    //multi_matmult<<<1, NUM_THREADS>>>(num_dim, d_mat_dim, d_mat_list, d_out_mat);
    //gpuErrchk(cudaPeekAtLastError());

    //gpu_seq_multi_matmult<<<1, NUM_THREADS>>>(num_dim, d_mat_dim, d_mat_list, d_out_mat1, d_out_mat2);
    //cudaThreadSynchronize();

/*
    int ** tmp = (int **) malloc((num_dim - 1) * sizeof(int *));
    cudaMemcpy(tmp, d_mat_list, num_mat * sizeof(int *), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_dim - 1; i++) {
        printf("%p\n", tmp[i]);
        cudaMemcpy(out_mat, tmp[i], mat_dim[i] * mat_dim[i+1] * sizeof(int), cudaMemcpyDeviceToHost);
        print_mat(out_mat, mat_dim[i], mat_dim[i+1]);

    }
*/
/*
    int iii[] = {0, 0, 0, 0, 0};
    int * d_iii;
    cudaMalloc((void **)&d_iii,5 *  sizeof(int));
    cudaMemcpy(d_iii, iii, 5 * sizeof(int), cudaMemcpyHostToDevice);
*/
    for (int i = 0; i < nblocks; i++) {

        printf("nblocks %d, i %d\n", nblocks, i);
        printf("mat_list_partition %d %d\n", mat_list_partition[i], mat_list_partition[i+1] - 1);
        printf("end mat_dim %d %d\n", mat_dim[mat_list_partition[i]], mat_dim[mat_list_partition[i+1]]);
        gpu_par_multi_matmult<<<1, NUM_THREADS>>>(mat_list_partition[i], mat_list_partition[i+1] + 1, d_mat_dim, d_mat_list, int_mat1[i], int_mat2[i]); //, d_iii);
        //cudaMemcpy(iii, d_iii, 5 * sizeof(int), cudaMemcpyDeviceToHost);
        //printf("num_products %d\n", iii[0]);
        //printf("num_rows %d\n", iii[1]);
        //printf("num_cols %d\n", iii[2]);
    }
    cudaDeviceSynchronize();

// Break up case for when only one block is chosen, and when many blocks (more than 1) is chosen

    //print_mat((int*)int_mat1, 1, nblocks);
    //print_mat((int*)int_mat2, 1, nblocks);
    for (int i = 0; i < nblocks; i++) {
        cudaMemcpy(out_mat, int_mat1[i], mat_dim[mat_list_partition[i]] * mat_dim[mat_list_partition[i+1]] * sizeof(int), cudaMemcpyDeviceToHost);
        printf("printing int_mat1[%d]\n", i);
        print_mat(out_mat, mat_dim[mat_list_partition[i]], mat_dim[mat_list_partition[i+1]]);
        cudaMemcpy(out_mat, int_mat2[i], mat_dim[mat_list_partition[i]] * mat_dim[mat_list_partition[i+1]] * sizeof(int), cudaMemcpyDeviceToHost);
        printf("printing int_mat2[%d]\n", i);
        print_mat(out_mat, mat_dim[mat_list_partition[i]], mat_dim[mat_list_partition[i+1]]);
        if ((mat_list_partition[i+1] - mat_list_partition[i]) % 2 == 1) {
            int_mat_final[i] = int_mat1[i];
        } else {
            int_mat_final[i] = int_mat2[i];
        }
    }
    //print_mat((int*)int_mat_final, 1, nblocks);
    //cudaMemcpy(out_mat, int_mat_final, mat_dim[mat_list_partition[i]] * mat_dim[mat_list_partition[i]] * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_int_mat_final, int_mat_final, nblocks * sizeof(int *), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    
    printf("Calling last kernel\n");
    gpu_par_multi_matmult<<<1, NUM_THREADS>>>(0, nblocks + 1, d_int_mat_dim, d_int_mat_final, d_int_output_mat1, d_int_output_mat2); //, iii);
    cudaDeviceSynchronize();


    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
       // print the CUDA error message and exit
       printf("CUDA error: %s\n", cudaGetErrorString(error));
       exit(-1);
    }

    if (nblocks % 2 == 1) {
        cudaMemcpy(out_mat, d_int_output_mat1, mat_dim[0] * mat_dim[num_dim-1] * sizeof(int), cudaMemcpyDeviceToHost);   
    } else {
        cudaMemcpy(out_mat, d_int_output_mat2, mat_dim[0] * mat_dim[num_dim-1] * sizeof(int), cudaMemcpyDeviceToHost);   
    }

    printf("%d %d\n", mat_dim[mat_list_partition[0]], mat_dim[mat_list_partition[nblocks]]);
    printf("gpu par\n");
    print_mat(out_mat, mat_dim[0], mat_dim[num_dim-1]);
    printf("\n");

    if_mats_equal(out_mat, cpu_mat, mat_dim[0], mat_dim[num_dim-1]);
/*    
    if (num_dim % 2 == 1) {
        cudaMemcpy(out_mat, d_out_mat1, mat_dim[0] * mat_dim[num_dim-1] * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(out_mat, d_out_mat2, mat_dim[0] * mat_dim[num_dim-1] * sizeof(int), cudaMemcpyDeviceToHost);
    }

    printf("gpu seq\n");
    print_mat(out_mat, mat_dim[0], mat_dim[num_dim-1]);
    printf("\n");

    if_mats_equal(out_mat, cpu_mat, mat_dim[0], mat_dim[num_dim-1]);
*/    
    return 0;

}
