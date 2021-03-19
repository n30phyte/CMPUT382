// Kernel for adding to vectors
// There are a few TODOs that you need to fill out
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
	int i = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (i < len) {
		out[i] = in1[i] + in2[i];
	}
}

int main(int argc, char **argv) {
	wbArg_t args;
	int inputLength;
	float *hostInput1;
	float *hostInput2;
	float *hostOutput;
	float *deviceInput1;
	float *deviceInput2;
	float *deviceOutput;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput1 =
		(float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	hostInput2 =
		(float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *)malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
	cudaMalloc((void**)&deviceInput1, inputLength * sizeof(float));
	cudaMalloc((void**)&deviceInput2, inputLength * sizeof(float));
	cudaMalloc((void**)&deviceOutput, inputLength * sizeof(float));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	int threadsPerBlock = 1024;
	int numBlocks = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

	wbTime_start(Compute, "Performing CUDA computation");
	vecAdd << <threadsPerBlock, numBlocks >> >(deviceInput1, deviceInput2, deviceOutput, inputLength);

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	cudaDeviceSynchronize();

	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}

