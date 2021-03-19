#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "libwb/wb.h"

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH / 2

__global__ void convolution(const float *__restrict__ I, const float *__restrict__ M, float *__restrict__ P,
                            const int channels, const int width, const int height) {
    const unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < width && j < height) {
        for (unsigned int c = 0; c < channels; c++) {
            float accumulator = 0;
            for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
                for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                    const int imgX = (int) i + x;
                    const int imgY = (int) j + y;
                    if ((imgX >= 0 && imgX < width) &&
                        (imgY >= 0 && imgY < height)) {
                        unsigned const int imgXY = (imgX + imgY * width) * channels + c;
                        const float imgPixel = I[imgXY];
                        const float maskValue = M[(x + MASK_RADIUS) + (y + MASK_RADIUS) * MASK_WIDTH];
                        accumulator += imgPixel * maskValue;
                    }
                }
            }
            P[(i + j * width) * channels + c] = __saturatef(accumulator);
        }
    }
}

int main(int argc, char *argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    char *inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *hostMaskData;
    float *deviceInputImageData;
    float *deviceOutputImageData;
    float *deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");

    const size_t imageSize = imageWidth * imageHeight * imageChannels * sizeof(float);
    const size_t maskSize = maskRows * maskColumns * sizeof(float);

    cudaMalloc((void **) &deviceInputImageData, imageSize);
    cudaMalloc((void **) &deviceOutputImageData, imageSize);

    cudaMalloc((void **) &deviceMaskData, maskSize);

    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");

    cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskSize, cudaMemcpyHostToDevice);

    wbTime_stop(Copy, "Copying data to the GPU");

    const int threads = 32;
    const dim3 blockSize(threads, threads, 1);
    const int gridX = (imageWidth + threads - 1) / threads;
    const int gridY = (imageHeight + threads - 1) / threads;
    const dim3 gridSize(gridX, gridY, 1);

    wbTime_start(Compute, "Doing the computation on the GPU");
    convolution<<<gridSize, blockSize>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceMaskData);
    cudaFree(deviceOutputImageData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

#if LAB_DEBUG
    system("pause");
#endif

    return 0;
}