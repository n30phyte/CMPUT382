#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "exclusive_scan.h"

__global__ void exclusiveScan(float* input, float* output, int N) {
    __shared__ float sharedInput[BLOCK_SIZE];
    int tx = threadIdx.x;
    int i = tx + blockIdx.x * blockDim.x;
    if(i < N && tx != 0) {
        sharedInput[tx] = input[i-1];
    } else {
        sharedInput[tx] = 0;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride <= threadIdx.x; stride <<= 1) {
        __syncthreads();
        sharedInput[tx] += sharedInput[tx - stride];
    }
    if(i < N) {
        output[i] = sharedInput[tx];
    }
}
