#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "wb.h"

#define TILE_WIDTH 16

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int col = bx * TILE_WIDTH + tx;
    unsigned int row = by * TILE_WIDTH + ty;

    double sum = 0;

    // Go through the phases, i is phase number.
    for (unsigned int i = 0; i < (TILE_WIDTH + numAColumns - 1) / TILE_WIDTH; i++) {

        // Load values
        if ((i * TILE_WIDTH + tx) < numAColumns && row < numCRows) {
            sharedA[ty][tx] = A[row * numAColumns + (i * TILE_WIDTH + tx)];
        } else {
            sharedA[ty][tx] = 0.0;
        }

        if ((i * TILE_WIDTH + ty) < numBRows && col < numCColumns) {
            sharedB[ty][tx] = B[(i * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            sharedB[ty][tx] = 0.0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int k = 0; k < TILE_WIDTH; k++) {
                sum += sharedA[ty][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = sum;
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    args = wbArg_read(argc, argv);

#if LAB_DEBUG
    std::cout << "Running Tiled Matrix Multiplicaion ..." << std::endl;
#endif

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA =
            (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB =
            (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    hostC = (float *) malloc(numARows * numBColumns * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    numCRows = numARows;
    numCColumns = numBColumns;

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    unsigned int threads = TILE_WIDTH;
    unsigned int blocksX = (numCColumns + threads - 1) / threads;
    unsigned int blocksY = (numCRows + threads - 1) / threads;

    dim3 blockSize(threads, threads, 1);
    dim3 gridSize(blocksX, blocksY, 1);

    wbLog(TRACE, "The block dimensions are ", blockSize.x, " x ", blockSize.y);
    wbLog(TRACE, "The grid dimensions are ", gridSize.x, " x ", gridSize.y);

    wbTime_start(Compute, "Performing CUDA computation");
    matrixMultiplyShared<<<gridSize, blockSize>>>(
            deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}