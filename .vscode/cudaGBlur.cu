#include <iostream>
#include <fstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <cstdint>

// CUDA header
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#pragma pack(push, 1) // Ensure structure alignment
struct BMPHeader {
    uint16_t fileType;
    uint32_t fileSize;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t dataOffset;
    uint32_t headerSize;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t totalColors;
    uint32_t importantColors;
};
#pragma pack(pop)

struct Pixel {
    unsigned char blue, green, red; // BMP stores pixel data in BGR order
};

// CUDA kernel to calculate the Gaussian weight for a given distance from the center
__device__ float gaussianWeight(int x, int y, float sigma) {
    return exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
}

// CUDA kernel to apply Gaussian blur to a single pixel
__global__ void gaussianBlurKernel(const Pixel* inputImage, Pixel* outputImage, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int size = static_cast<int>(ceil(3 * sigma));
        int center = size / 2;

        float totalWeight = 0.0;
        float totalRed = 0.0, totalGreen = 0.0, totalBlue = 0.0;

        for (int i = -center; i <= center; ++i) {
            for (int j = -center; j <= center; ++j) {
                int newX = x + i;
                int newY = y + j;

                // Ensure that the indices are within bounds
                if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                    float weight = gaussianWeight(i, j, sigma);
                    totalRed += weight * inputImage[newX + newY * width].red;
                    totalGreen += weight * inputImage[newX + newY * width].green;
                    totalBlue += weight * inputImage[newX + newY * width].blue;
                    totalWeight += weight;
                }
            }
        }

        outputImage[y * width + x].red = static_cast<unsigned char>(totalRed / totalWeight);
        outputImage[y * width + x].green = static_cast<unsigned char>(totalGreen / totalWeight);
        outputImage[y * width + x].blue = static_cast<unsigned char>(totalBlue / totalWeight);
    }
}

// Function to apply Gaussian blur using CUDA
void applyGaussianBlurCUDA(std::vector<Pixel>& image, int width, int height, float sigma) {
    // Allocate GPU memory
    Pixel* d_inputImage, * d_outputImage;
    cudaMalloc((void**)&d_inputImage, width * height * sizeof(Pixel));
    cudaMalloc((void**)&d_outputImage, width * height * sizeof(Pixel));

    // Copy input image to GPU
    cudaMemcpy(d_inputImage, image.data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    gaussianBlurKernel << <gridSize, blockSize >> > (d_inputImage, d_outputImage, width, height, sigma);

    // Check for kernel launch errors
    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }

    // Copy the result back to the CPU
    cudaMemcpy(image.data(), d_outputImage, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost);

    // Check for memory copy errors
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA memory copy error: " << cudaGetErrorString(cudaError) << std::endl;
        return;
    }

    // Free GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

int main() {
    std::string inputFilePath, outputFilePath;
    float sigma;

    std::cout << "Enter the input BMP image file path: ";
    std::cin >> inputFilePath;

    std::cout << "Enter sigma: ";
    std::cin >> sigma;

    std::cout << "Enter the output BMP image file path: ";
    std::cin >> outputFilePath;

    std::ifstream inputFile(inputFilePath, std::ios::binary);

    if (!inputFile) {
        std::cerr << "Error: Couldn't open the input BMP image file." << std::endl;
        return -1;
    }

    // Read BMP header
    BMPHeader bmpHeader;
    inputFile.read(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader));

    // Check if the file is a valid BMP file
    if (bmpHeader.fileType != 0x4D42 || bmpHeader.bitsPerPixel != 24) {
        std::cerr << "Error: Not a valid BMP file or unsupported color depth." << std::endl;
        inputFile.close();
        return -1;
    }

    // Move to the beginning of the pixel data
    inputFile.seekg(bmpHeader.dataOffset, std::ios::beg);

    // Read BMP pixel data
    std::vector<Pixel> image(bmpHeader.width * bmpHeader.height);

    // BMP data is stored upside down, so we read the rows in reverse order
    for (int y = bmpHeader.height - 1; y >= 0; --y) {
        for (int x = 0; x < bmpHeader.width; ++x) {
            inputFile.read(reinterpret_cast<char*>(&image[y * bmpHeader.width + x]), sizeof(Pixel));
        }

        // Skip any padding (if present) to align rows to multiples of 4 bytes
        inputFile.seekg((4 - (bmpHeader.width * 3) % 4) % 4, std::ios::cur);
    }

    // Close the input file
    inputFile.close();

    // Apply Gaussian blur to the image using CUDA
    applyGaussianBlurCUDA(image, bmpHeader.width, bmpHeader.height, sigma);

    // Write the blurred image to a new file
    std::ofstream outputFile(outputFilePath, std::ios::binary);

    if (!outputFile) {
        std::cerr << "Error: Couldn't create the output BMP image file." << std::endl;
        return -1;
    }

    // Write BMP header to the output file
    outputFile.write(reinterpret_cast<char*>(&bmpHeader), sizeof(BMPHeader));

    // Write blurred image pixel data to the output file
    for (int y = bmpHeader.height - 1; y >= 0; --y) {
        for (int x = 0; x < bmpHeader.width; ++x) {
            outputFile.write(reinterpret_cast<char*>(&image[y * bmpHeader.width + x]), sizeof(Pixel));
        }

        // Add any needed padding to align rows to multiples of 4 bytes
        for (int p = 0; p < (4 - (bmpHeader.width * 3) % 4) % 4; ++p) {
            outputFile.put(0);
        }
    }

    // Close the output file
    outputFile.close();

    std::cout << "BMP image blurred and saved successfully." << std::endl;

    return 0;
}
