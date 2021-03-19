#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "wb.h"

#define NUM_BINS 4096

__global__ void calculate_histogram(unsigned int *inputValues, unsigned int *outputBins, unsigned int inputLength) {


    __shared__ unsigned int histogram_private[NUM_BINS];
    for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        histogram_private[i] = 0;
    }
    __syncthreads();

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    while(idx < inputLength) {
        unsigned int bin = inputValues[idx];
        atomicAdd(&(histogram_private[bin]), 1);
        idx += stride;
    }
    __syncthreads();

    for (unsigned int bin_idx = threadIdx.x; bin_idx < NUM_BINS; bin_idx += blockDim.x) {
        atomicAdd(&(outputBins[bin_idx]), histogram_private[bin_idx]);
    }

}

__global__ void clamp_results(unsigned int *outputBins) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NUM_BINS) {
        outputBins[idx] = min(127, outputBins[idx]) == 127 ? 127 : outputBins[idx];
    }
}

#define CUDA_CHECK(ans)                                                   \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

int main(int argc, char *argv[]) {
    wbArg_t args;
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (unsigned int *) wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
    hostBins = (unsigned int *) malloc(NUM_BINS * sizeof(unsigned int));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);
    wbLog(TRACE, "The number of bins is ", NUM_BINS);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc((void **) &deviceBins, NUM_BINS * sizeof(unsigned int));
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    int threads = 512;
    dim3 gatherBlocks(64, 1, 1);
    dim3 cleanupBlocks((NUM_BINS + threads - 1) / threads, 1, 1);

    wbLog(TRACE, "Launching kernel");
    wbTime_start(Compute, "Performing CUDA computation");
    calculate_histogram<<<gatherBlocks, threads>>>(deviceInput, deviceBins, inputLength);
    cudaDeviceSynchronize();
    clamp_results<<<cleanupBlocks, threads>>>(deviceBins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceBins);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostBins, NUM_BINS);

    free(hostBins);
    free(hostInput);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}
