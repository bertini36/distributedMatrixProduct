/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* MULTI-NODE MATRIX-MATRIX PRODUCT WITH MPI                                 */
/*                                                                           */
/* File:         mmpmpi.cu                                                   */
/* Author:       Alberto Pou Quirós (Github: bertini36)                      */ 
/* Description:  This program performs a matrix product (A * B = C)          */
/*               distributing the computation between multiple nodes         */
/*               with MPI technology                                         */
/* Compilation:  mpicc -g -Wall -o mmpmpi mmpmpi.c                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1024

struct timeval start_time, end_time;

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

void showResults(double matrix_a[N][N], double matrix_b[N][N], double matrix_c[N][N]) {
    int i, j;
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
    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            (j % N == N-1) ? printf("%.1f \n", matrix_c[i][j]) : printf("%.1f,", matrix_c[i][j]);
        }
    }
}

int main(int argc, char *argv[]) {

    double A[N][N], B[N][N], C[N][N];
    int my_rank, comm_sz, from, to, i, j, k;
  
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

    if (my_rank == 0) {
        srand(time(NULL));
        initializeMatrices(A, B);
    }

    // Send necessary information to every process
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, N*N/comm_sz, MPI_DOUBLE, A[from], N*N/comm_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) { gettimeofday(&start_time, NULL); }

    // Calculate the product
    for (i=from; i<to; i++) 
        for (j=0; j<N; j++) {
            C[i][j] = 0;
        for (k=0; k<N; k++)
            C[i][j] += A[i][k] * B[k][j];
    }

    // Calculate compute time
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) { 
        gettimeofday(&end_time, NULL);
        printf("Compute time: %.1f ms \n", (float) (end_time.tv_sec - start_time.tv_sec) * 1000 + (end_time.tv_usec - start_time.tv_usec) / 1000);
     }

    // Receive results from every process
    MPI_Gather(C[from], N*N/comm_sz, MPI_DOUBLE, C, N*N/comm_sz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // if (my_rank == 0) { showResults(A, B, C); }

    MPI_Finalize();
    return 0;
}
