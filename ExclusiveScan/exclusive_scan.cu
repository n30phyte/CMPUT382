#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "exclusive_scan.h"

void recursiveScan(float* input, float* output, int numInputs) {
    const int scanGridSize = (numInputs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* aux;
    wbCheck(cudaMalloc((void **) &aux, scanGridSize * sizeof(float)));

    if(scanGridSize == 1) {
        exclusiveScan<<<scanGridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(input, output, NULL, numInputs);
        wbCheck(cudaDeviceSynchronize());
        cudaFree(aux);
        return;
    } else {
        float* scannedAux;
        cudaMalloc((void **) &scannedAux, scanGridSize * sizeof(float));
        exclusiveScan<<<scanGridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(input, output, aux, numInputs);
        recursiveScan(aux, scannedAux, scanGridSize);
        auxMerge<<<1, BLOCK_SIZE>>>(scannedAux, output, numInputs);
        wbCheck(cudaDeviceSynchronize());
        cudaFree(scannedAux);
        cudaFree(aux);
    }
}

__global__ void exclusiveScan(const float *input, float *output, float *S, int N) {
    extern __shared__ float sharedInput[];

    unsigned int tx = threadIdx.x;
    int i = tx + blockIdx.x * blockDim.x;

    if (i < N && i != 0) {
        sharedInput[tx] = input[i - 1];
    } else {
        sharedInput[tx] = 0;
    }

    // Down phase
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        __syncthreads();

        int idx = (tx + 1) * 2 * stride - 1;
        if (idx < blockDim.x) {
            sharedInput[idx] += sharedInput[idx - stride];
        }
    }

    // Up phase
    for (int stride = blockDim.x / 4; stride > 0; stride >>= 1) {
        __syncthreads();

        int idx = (tx + 1) * 2 * stride - 1;

        if (idx + stride < blockDim.x) {
            sharedInput[idx + stride] += sharedInput[idx];
        }
    }
    __syncthreads();


    if (i < N) {
        output[i] = sharedInput[tx];
        if (S != NULL && tx == (BLOCK_SIZE - 1)) {
            S[blockIdx.x] = sharedInput[tx];
        }
    }
}

__global__ void auxMerge(const float *offsets, float *input, int N) {
    const unsigned int tx = threadIdx.x;
    unsigned int startIdx = tx * BLOCK_SIZE;

        for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
            unsigned int idx = i + startIdx;
            if (idx < N) {
                input[idx] += offsets[tx];
            }
        }
}
