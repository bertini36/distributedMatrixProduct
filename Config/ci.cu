/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* CUDA INFORMATION                                                          */
/*                                                                           */
/* File:         cuda_information.cu                                         */
/* Description:  This program gets the important information of the          */
/*               available architecture                                      */
/* Compilation:  nvcc ci.cu -o ci                                            */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>

int main() {
    const int kb = 1024;
    const int mb = kb * kb;

    printf("CUDA version: %d v \n", CUDART_VERSION);    

    int devCount;
    cudaGetDeviceCount(&devCount);

    for (int i=0; i<devCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("%d : %s : %d.%d \n", i, props.name, props.major, props.minor);
        printf("Global memory: %d mb \n", props.totalGlobalMem / mb);
        printf("Shared memory: %d kb \n", props.sharedMemPerBlock / kb);
        printf("Constant memory: %d kb \n", props.totalConstMem / kb);
        printf("Block registers: %d \n", props.regsPerBlock);
        printf("Warp size: %d \n", props.warpSize);
        printf("Threads per block: %d \n", props.maxThreadsPerBlock);
        printf("Max block dimensions: [%d, %d, %d] \n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
        printf("Max grid dimensions:  [%d, %d, %d] \n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
        printf("Multiprocessor count: %d \n", props.multiProcessorCount);
        printf("Max threads per multiprocessor: %d \n", props.maxThreadsPerMultiProcessor);
    }

    return 0;
}