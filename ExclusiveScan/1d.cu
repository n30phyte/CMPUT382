#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "wb.h"

#include "exclusive_scan.h"

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    float *auxScanBuffer;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    cudaHostAlloc(&hostOutput, numElements * sizeof(float),
                  cudaHostAllocDefault);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
          numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceInput, numElements * sizeof(float));
    cudaMalloc((void **) &deviceOutput, numElements * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    cudaMemset(deviceOutput, 0, numElements * sizeof(float));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");

    const int scanGridSize = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc((void **) &auxScanBuffer, scanGridSize * sizeof(float));
    recursiveScan(deviceInput, deviceOutput, numElements);
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    cudaFreeHost(hostOutput);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}
