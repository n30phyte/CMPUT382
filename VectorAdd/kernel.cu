#include "kernel.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vecAdd(const float *in1, const float *in2, float *out, const int len) {
    unsigned int i = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

void addVectors(std::vector<float> &input1, std::vector<float> &input2, std::vector<float> &output, int inputLength) {
    float *deviceInput1;
    float *deviceInput2;
    float *deviceOutput;

    cudaMalloc((void **) &deviceInput1, inputLength * sizeof(float));
    cudaMalloc((void **) &deviceInput2, inputLength * sizeof(float));
    cudaMalloc((void **) &deviceOutput, inputLength * sizeof(float));

    cudaMemcpy(deviceInput1, input1.data(), inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, input2.data(), inputLength * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int numBlocks = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

    vecAdd <<< threadsPerBlock, numBlocks >>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data(), deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
}
