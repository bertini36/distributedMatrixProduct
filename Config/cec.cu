/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* CUDA EXECUTION CONFIGURATION                                              */
/* 											                                 */
/* File:    	 cuda_execution_configuration.cu                             */
/* Description:  This program suggests which is the best configuration of    */
/*				 CUDA blocks based on the available architecture             */
/*				 parallelizing the computation with Nvidia CUDA technology   */
/* Compilation:  nvcc cec.cu -o cec                                          */  
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define min(a, b) ((a < b) ? a : b )
#define max(a, b) ((a > b) ? a : b )

//−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
// In:  numberOfThreads, registersPerThread, sharedPerThread
// Out: bestThreadsPerBlock, bestTotalBlocks
void calcExecConf(int numberOfThreads, int registersPerThread, int sharedPerThread, int &bestThreadsPerBlock, int &bestTotalBlocks) {

	cudaDeviceProp pr;
	cudaGetDeviceProperties(&pr, 0); // Replace 0 with appropriate ID in case of a multi−GPU system

	int maxRegs = pr.regsPerBlock;
	int SM = pr.multiProcessorCount;
	int warp = pr.warpSize;
	int sharedMem = pr.sharedMemPerBlock;
	int maxThreadsPerSM = pr.maxThreadsPerMultiProcessor;
	int totalBlocks;
	float imbalance, bestimbalance;
	int threadsPerBlock;

	int numWarpSchedulers;
	switch (pr.major) {
		case 1:
			numWarpSchedulers = 1;
			break;
		case 2:
			numWarpSchedulers = 2;
			break;
		default:
			numWarpSchedulers = 4;
			break;
	}

	bestimbalance = SM;

	// Initially calculate the maximum possible threads per block. Incorporate limits imposed by:
	// 1) SM hardware
	threadsPerBlock = maxThreadsPerSM;
	// 2) registers
	threadsPerBlock = min(threadsPerBlock, maxRegs/registersPerThread);
	// 3) shared memory size
	threadsPerBlock = min(threadsPerBlock, sharedMem/sharedPerThread);
	// 6.7 Optimization techniques
	// Make sure it is a multiple of warp Size
	int tmp = threadsPerBlock / warp;

	for (threadsPerBlock = tmp ∗ warp; threadsPerBlock >= numWarpSchedulers ∗ warp && bestimbalance != 0; threadsPerBlock −= warp) {
		totalBlocks = (int) ceil(1.0 ∗ numberOfThreads / threadsPerBlock);
		if (totalBlocks % SM == 0)
			imbalance = 0;
		else {
			int blocksPerSM = totalBlocks / SM; 
			imbalance = (SM − (totalBlocks % SM)) / (blocksPerSM + 1.0);
		}
		if (bestimbalance >= imbalance) {
			bestimbalance = imbalance;
			bestThreadsPerBlock = threadsPerBlock;
			bestTotalBlocks = totalBlocks;
		}
	}
}

int main() {
	int bestThreadsPerBlock, bestTotalBlocks;
	calcExecConf(64 * 64, 2, 2, &bestThreadsPerBlock, &bestTotalBlocks);
	printf("Best threads per block: %d \n", bestThreadsPerBlock);
	printf("Best total blocks: %d \n", bestTotalBlocks);
	return 0;
} 