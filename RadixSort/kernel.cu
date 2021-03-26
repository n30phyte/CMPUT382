#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "wb.h"

#define BLOCK_SIZE 512 //TODO: You can change this

#define wbCheck(ans) gpuAssert((ans), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void exclusiveScan(const int *input, int *output, int *S, int N) {
    extern __shared__ int sharedInput[];

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
        if (S != nullptr && tx == (BLOCK_SIZE - 1)) {
            S[blockIdx.x] = sharedInput[tx];
        }
    }
}

__global__ void auxMerge(const int *offsets, int *input, int N) {
    const unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int startIdx = tx * BLOCK_SIZE;

    for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
        unsigned int idx = i + startIdx;
        if (idx < N) {
            input[idx] += offsets[tx];
        }
    }
}

void recursiveScan(int *input, int *output, int numInputs) {
    const int scanGridSize = (numInputs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (scanGridSize == 1) {
        exclusiveScan<<<scanGridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, nullptr, numInputs);
        wbCheck(cudaDeviceSynchronize());
        return;
    } else {
        int *aux;
        int *scannedAux;
        wbCheck(cudaMalloc((void **) &aux, scanGridSize * sizeof(int)));
        wbCheck(cudaMalloc((void **) &scannedAux, scanGridSize * sizeof(int)));

        exclusiveScan<<<scanGridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(input, output, aux, numInputs);
        wbCheck(cudaDeviceSynchronize());
        recursiveScan(aux, scannedAux, scanGridSize);

        int mergeGrids = (scanGridSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auxMerge<<<mergeGrids, BLOCK_SIZE>>>(scannedAux, output, numInputs);
        wbCheck(cudaDeviceSynchronize());

        cudaFree(scannedAux);
        cudaFree(aux);
    }
}

__global__ void checkBits(const int *__restrict__ input, int *__restrict__ output, const int N, const int radix) {
    extern __shared__ int sharedInput[];

    unsigned int tx = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        sharedInput[tx] = input[idx];
        output[idx] = ~(sharedInput[tx] >> radix) & 1;
    }
}

__global__ void scatter(const int *__restrict__ input, const int *__restrict__ bitArray, const int *__restrict__ scannedBits, int *__restrict__ output, const int N) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        unsigned int totalFalses = bitArray[N - 1] + scannedBits[N - 1];
        unsigned int target = idx - scannedBits[idx] + totalFalses;
        unsigned int destination = bitArray[idx] ? scannedBits[idx] : target;
        output[destination] = input[idx];
    }
}

__global__ void copyMemory(const int *__restrict__ input, int *__restrict output, const int N) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        output[idx] = input[idx];
    }
}

void sort(int *d_deviceInput, int *d_deviceOutput, int numElements) {
    int *bitArray;
    int *scannedBits;

    cudaMalloc((void **) &bitArray, numElements * sizeof(int));
    cudaMalloc((void **) &scannedBits, numElements * sizeof(int));

    dim3 blockSize(BLOCK_SIZE, 1, 1);
    dim3 gridSize((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    cudaStream_t stream;
    for (int i = 0; i < 16; i++) {
        checkBits<<<gridSize, blockSize, BLOCK_SIZE * sizeof(int)>>>(d_deviceInput, bitArray, numElements, i);
        cudaDeviceSynchronize();
        recursiveScan(bitArray, scannedBits, numElements);
        scatter<<<gridSize, blockSize>>>(d_deviceInput, bitArray, scannedBits, d_deviceOutput, numElements);
        cudaDeviceSynchronize();
        if (i != 15) {
            copyMemory<<<gridSize, blockSize>>>(d_deviceOutput, d_deviceInput, numElements);
        }
    }

    cudaFree(bitArray);
    cudaFree(scannedBits);
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
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(int), cudaMemcpyDeviceToHost));
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
