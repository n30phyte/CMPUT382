#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define BLOCK_SIZE 512 //TODO: You can change this

#define wbCheck(ans) gpuAssert((ans), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void sort(int *d_deviceInput, int *d_deviceOutput, int numElements) {
    //TODO: Modify this to complete the functionality of the sort on the deivce

}

int main(int argc, char **argv) {
    wbArg_t args;
    int *hostInput;  // The input 1D list
    int *hostOutput; // The output list
    int *deviceInput;
    int *deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (int *) wbImport(wbArg_getInputFile(args, 0), &numElements, "integral_vector");
    cudaHostAlloc(&hostOutput, numElements * sizeof(int), cudaHostAllocDefault);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void **) &deviceInput, numElements * sizeof(int)));
    wbCheck(cudaMalloc((void **) &deviceOutput, numElements * sizeof(int)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(int)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(int),
                       cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");
    sort(deviceInput, deviceOutput, numElements);
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                       cudaMemcpyDeviceToHost));
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
