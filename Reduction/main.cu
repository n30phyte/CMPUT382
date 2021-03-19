#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "libwb/wb.h"

#define BLOCK_SIZE 512

__global__ void total(const float *input, float *output, unsigned int len) {
    __shared__ float shared_data[BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tx;

    shared_data[tx] = (i < len) ? input[i] : 0.0f;
    shared_data[tx] += (i + blockDim.x < len) ? input[i + blockDim.x] : 0.0f;

    for (unsigned int slice = blockDim.x >> 1; slice > 0; slice >>= 1) {
        if (tx < slice) {
            shared_data[tx] += input[i + blockDim.x];
        }
        __syncthreads();
    }

    if (tx == 0) {
        output[blockIdx.x] = shared_data[tx];
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numInputElements;  // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0),
                                   &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE << 1);
    if (numInputElements % (BLOCK_SIZE << 1)) {
        numOutputElements++;
    }
    hostOutput = (float *) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
          numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ",
          numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceInput, numInputElements * sizeof(float));
    cudaMalloc((void **) &deviceOutput, numOutputElements * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float),
               cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    int blockSize = BLOCK_SIZE;
    int gridSize = (numInputElements + (blockSize * 2 - 1)) / (blockSize * 2);

    wbTime_start(Compute, "Performing CUDA computation");

    total<<<gridSize, blockSize>>>(deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float),
               cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}