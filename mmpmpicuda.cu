/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* DISTRIBUTED AND PARALLEL MATRIX-MATRIX PRODUCT WITH MPI AND CUDA          */
/*                                                                           */
/* File:         mmpmpicuda.cu                                               */
/* Author:       Alberto Pou Quirós (Github: bertini36)                      */ 
/* Description:  This program performs a matrix product (A * B = C)          */
/*               distributing the computation between multiple nodes         */
/*               with MPI technology and parallelizing the computation in    */
/*               every node with Nvidia CUDA technology                      */
/* Compilation:  nvcc -I/opt/mpi/bullxmpi/1.2.9.1/include                    */  
/*               -L/opt/mpi/bullxmpi/1.2.9.1/lib -lmpi -ldl -lm -lnuma       */
/*               -lrt -lnsl -lutil -lm -ldl mmpmpicuda.cu -o mmpmpicuda      */
/* Strategy:                                                                 */
/*                  Example 16x16 matrices with 4 nodes:                     */
/*                   _________________16________________                     */
/*                   |                                 |                     */
/*                   |               NODE 1            | 4                   */
/*                   |_________________________________|                     */
/*                   |                                 |                     */
/*                   |               NODE 2            | 4                   */
/*              C =  |_________________________________|    16               */ 
/*                   |                                 |                     */
/*                   |               NODE 3            | 4                   */
/*                   |_________________________________|                     */
/*                   |                                 |                     */ 
/*                   |               NODE 4            | 4                   */
/*                   |_________________________________|                     */
/*                                                                           */
/*                  Node 1 computes 4 rows of result matrix:                 */
/*                   __________________________________                      */
/*                   |                                 |                     */
/*                   |         4x16 CUDA block         |                     */
/*                   |_________________________________|                     */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

#define N 1024

#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

struct timeval start_time, end_time;

inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

__global__ void matrixProduct(double *matrix_a, double *matrix_b, double *matrix_c, int width, int nrows, int my_rank) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x; 
    matrix_c[row * width + col] = 0;
    for (int k=0; k<width; k++) {
        matrix_c[row * width + col] += matrix_a[(row * width) + (my_rank * nrows * width) + k] * matrix_b[k * width + col];
    }
}

void initializeMatrices(double matrix_a[N][N], double matrix_b[N][N]) {
    int i, j;
    srand(time(NULL));
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            matrix_a[i][j] = rand();
            matrix_b[i][j] = rand();
        }
    }
}

void showMatrices(double matrix_a[N][N], double matrix_b[N][N], double matrix_c[N][N]) {
    int i, j;
    srand(time(NULL));
    printf("***** MATRIX A ***** \n");
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            (j % N == N-1) ? printf("%.1f \n", matrix_a[i][j]) : printf("%.1f,", matrix_a[i][j]);
        }
    }
    printf("***** MATRIX B ***** \n");
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            (j % N == N-1) ? printf("%.1f \n", matrix_b[i][j]) : printf("%.1f,", matrix_b[i][j]);
        }
    }
    printf("***** RESULT MATRIX ***** \n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            (j % N == N-1) ? printf("%f \n", matrix_c[i][j]) : printf("%f,", matrix_c[i][j]);
        }
    }
}

int main(int argc, char *argv[]) {

    double A[N][N], B[N][N], C[N][N];
    double *d_a, *d_b, *d_c;
    int my_rank, comm_sz, from, to, nrows;
  
    // MPI initialization
    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);    // Process id 
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);    // Number of processors 

    if (N % comm_sz != 0) {
        if (my_rank == 0) printf("Matrix size not divisible by number of processors \n");
        MPI_Finalize();
        exit(-1);
    }

    // Calculate interval lines to compute per node
    from = my_rank * N / comm_sz;
    to = (my_rank + 1) * N / comm_sz;
    nrows = to - from;

    if (my_rank == 0) { initializeMatrices(A, B); }

    // Send A y B to every node
    MPI_Bcast(A, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory in the device
    checkCuda(cudaMalloc((void **) &d_a, N*N*sizeof(double)));
    checkCuda(cudaMalloc((void **) &d_b, N*N*sizeof(double)));
    checkCuda(cudaMalloc((void **) &d_c, (N*N/comm_sz)*sizeof(double)));

    // Copy the information in the device
    checkCuda(cudaMemcpy(d_a, A, N*N*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, B, N*N*sizeof(double), cudaMemcpyHostToDevice));

    // CUDA threads structure definition
    dim3 dimGrid(1);
    dim3 dimBlock(N, nrows);    

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) { gettimeofday(&start_time, NULL); }

    // Kernel launch
    matrixProduct<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N, nrows, my_rank);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());

    // Calculate compute time
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) { 
        gettimeofday(&end_time, NULL);
        printf("Compute time: %.1f ms \n", (float) (end_time.tv_sec - start_time.tv_sec) * 1000 + (end_time.tv_usec - start_time.tv_usec) / 1000);
     }

    // Get results from device
    checkCuda(cudaMemcpy(C[from], d_c, (nrows)*N*sizeof(double), cudaMemcpyDeviceToHost));

    // Unify results from nodes
    MPI_Gather(C[from], N*N/comm_sz, MPI_DOUBLE, C, N*N/comm_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // if (my_rank == 0)  { showMatrices(A, B, C); }

    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));
    
    MPI_Finalize();
    return 0;

}