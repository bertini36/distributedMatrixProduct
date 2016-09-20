/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* PARALLEL MATRIX-MATRIX PRODUCT WITH CUDA                                  */
/*                                                                           */
/* File:         mmpcuda.cu                                                  */
/* Author:       Alberto Pou Quirós (Github: bertini36)                      */ 
/* Description:  This program performs a matrix product (A * B = C)          */
/*               parallelizing the computation with Nvidia CUDA technology   */
/* Compilation:  nvcc mmpcuda.cu -o mmpcuda                                  */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 1024
#define BLOCK_SIZE_DIM 16 

#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

__global__ void matrixProduct(double *matrix_a, double *matrix_b, double *matrix_c, int width) {
    int sum = 0;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (col < width && row < width) {
        for (int k=0; k<width; k++) {
            sum += matrix_a[row * width + k] * matrix_b[k * width + col];
        }
        matrix_c[row * width + col] = sum;
    }
}

void initializeMatrices(double matrix_a[N][N], double matrix_b[N][N]) {
    srand(time(NULL));
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            matrix_a[i][j] = rand();
            matrix_b[i][j] = rand();
        }
    }
}

void showResults(double matrix_a[N][N], double matrix_b[N][N], double matrix_c[N][N]) {
    printf("***** MATRIX A ***** \n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            (j % N == N-1) ? printf("%.1f \n", matrix_a[i][j]) : printf("%.1f,", matrix_a[i][j]);
        }
    }
    printf("***** MATRIX B ***** \n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            (j % N == N-1) ? printf("%.1f \n", matrix_b[i][j]) : printf("%.1f,", matrix_b[i][j]);
        }
    }
    printf("***** RESULT MATRIX ***** \n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            (j % N == N-1) ? printf("%.1f \n", matrix_c[i][j]) : printf("%.1f,", matrix_c[i][j]);
        }
    }
}

int main() {

    double h_a[N][N], h_b[N][N], h_c[N][N];
    double *d_a, *d_b, *d_c;

    initializeMatrices(h_a, h_b);

    double size = (double) N * N * sizeof(double);

    // Allocate memory in the device
    checkCuda(cudaMalloc((void **) &d_a, size));
    checkCuda(cudaMalloc((void **) &d_b, size));
    checkCuda(cudaMalloc((void **) &d_c, size));

    // Copy the information in the device
    checkCuda(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // CUDA threads structure definition
    dim3 dimGrid((N + BLOCK_SIZE_DIM -1) / BLOCK_SIZE_DIM, (N + BLOCK_SIZE_DIM -1) / BLOCK_SIZE_DIM);
    dim3 dimBlock(BLOCK_SIZE_DIM, BLOCK_SIZE_DIM);  

    // Create events
    cudaEvent_t event1, event2;
    checkCuda(cudaEventCreate(&event1));
    checkCuda(cudaEventCreate(&event2));

    // Record events around kernel launch
    checkCuda(cudaEventRecord(event1, 0)); 
    matrixProduct<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
    checkCuda(cudaEventRecord(event2, 0));

    // Synchronize
    checkCuda(cudaEventSynchronize(event1));
    checkCuda(cudaEventSynchronize(event2)); // Wait for the event to be executed!

    // Calculate compute time
    float dt_ms;
    checkCuda(cudaEventElapsedTime(&dt_ms, event1, event2));
    printf("Compute time: %.1f ms \n", dt_ms);

    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());

    // Copy results from device to the host
    checkCuda(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));

    // showResults(h_a, h_b, h_c);

    cudaDeviceReset();
    return 0;
}