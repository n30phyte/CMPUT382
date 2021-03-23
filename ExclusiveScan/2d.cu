#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

#include "wb.h"
#include "exclusive_scan.h"

#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8

__global__ void kernelPrint(float *array, int cols, int rows) {
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            printf("%f ", array[j * cols + i]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void transposeMatrix(const float *input, float *output, int width, int height) {
    __shared__ float block[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    unsigned int xIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = input[index_in];
    }
    __syncthreads();

    xIndex = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;
        output[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void transposeCoalesced(const float *idata, float *odata) {
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;
    int width = gridDim.x * TRANSPOSE_TILE_DIM;

    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];

}

int main(int argc, char **argv) {

    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;  // device input
    float *deviceTmpOutput;  // temporary output
    float *deviceOutput;  // output
    int numInputRows, numInputCols; // dimensions of the array

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputRows, &numInputCols);
    cudaHostAlloc(&hostOutput, numInputRows * numInputCols * sizeof(float), cudaHostAllocDefault);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of input are ",
          numInputRows, "x", numInputCols);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceInput, numInputRows * numInputCols * sizeof(float));
    cudaMalloc((void **) &deviceOutput, numInputRows * numInputCols * sizeof(float));
    cudaMalloc((void **) &deviceTmpOutput, numInputRows * numInputCols * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    cudaMemset(deviceOutput, 0, numInputRows * numInputCols * sizeof(float));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceInput, hostInput, numInputRows * numInputCols * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");

    for (int i = 0; i < numInputRows; ++i) {
        recursiveScan(&deviceInput[numInputCols * i], &deviceTmpOutput[numInputCols * i], numInputCols);
    }

//    kernelPrint<<<1, 1>>>(deviceTmpOutput, numInputCols, numInputRows);
//    wbCheck(cudaDeviceSynchronize());

    dim3 transposeBlockDim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
    dim3 transposeGridDim(ceil(numInputCols / (float) TRANSPOSE_TILE_DIM), ceil(numInputRows / (float) TRANSPOSE_TILE_DIM));
//    transposeCoalesced<<<transposeGridDim, transposeBlockDim>>>(deviceTmpOutput, deviceOutput, numInputCols, numInputRows);
    transposeCoalesced<<<transposeGridDim, transposeBlockDim>>>(deviceTmpOutput, deviceOutput);
    wbCheck(cudaDeviceSynchronize());

    for (int i = 0; i < numInputCols; ++i) {
        recursiveScan(&deviceOutput[numInputRows * i], &deviceTmpOutput[numInputRows * i], numInputRows);
    }

    // You can change TranposeBlockDim and TranposeGridDim, but if you use kernel suggested in the manual file, these should be the correct ones
    transposeBlockDim = dim3(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
    transposeGridDim = dim3(ceil(numInputRows / (float) TRANSPOSE_TILE_DIM), ceil(numInputCols / (float) TRANSPOSE_TILE_DIM));
//    transposeCoalesced<<<transposeGridDim, transposeBlockDim>>>(deviceTmpOutput, deviceOutput, numInputRows, numInputCols);
    transposeCoalesced<<<transposeGridDim, transposeBlockDim>>>(deviceTmpOutput, deviceOutput);
    wbCheck(cudaDeviceSynchronize());

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, numInputRows * numInputCols * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceTmpOutput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numInputRows, numInputCols);

    free(hostInput);
    cudaFreeHost(hostOutput);

    cudaDeviceSynchronize();

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}
