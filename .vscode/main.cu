#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <inttypes.h>
#include <iterator>

// 240p : 426 width 240 height
// 360p : 640 width 360 height
// 480p : 854 width 480 height
// 720p : 1280 width 720 height
// 1080p : 1920 width 1080 height

//1024 threads per block
// 2^32 - 1 x 65,535 x 65,535 in (x,y,z)

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

typedef struct Pixel { // BMP stores pixel data in BGR order
    unsigned char blue;
    unsigned char green;
    unsigned char red; 
    unsigned char reserved;
}Pixel;

__host__ __device__ void setPixel(unsigned char * image, int width, int height, Pixel* pixel)
{
    image[0] = pixel->blue;
    image[1] = pixel->green;
    image[2] = pixel->red;
    image[3] = pixel->reserved;
}

__host__ __device__ void getPixel(unsigned char * image, int width, int height, Pixel* pixel)
{
    pixel->blue = image[0];
    pixel->green = image[1];
    pixel->red = image[2];
    pixel->reserved = image[3];
}

__global__ void averageBlur(unsigned char* image_in, unsigned char* image_out, int width, int height, int radius)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int x_dir = (x / width);
    int y_dir = (y / height);

    int totalPixels = 0;
    int totalRed = 0, totalGreen = 0, totalBlue = 0;
    Pixel inputPixel, outputPixel;
    for (int i = -radius; i < radius; i++)
    {
        for (int j = -radius; j < radius; j++)
        {
            int newWidth = x_dir + i;
            int newHeight = y_dir + j;
            if (newWidth >= 0 && newWidth < width && newHeight >= 0 && newHeight < height)
            {
                getPixel(image_in, newWidth, newHeight, &inputPixel);
                totalRed += inputPixel.red;
                totalGreen += inputPixel.green;
                totalBlue += inputPixel.blue;
                ++totalPixels;
            }
        }
    }
    outputPixel.red = static_cast<unsigned char>(totalRed / totalPixels);
    outputPixel.green = static_cast<unsigned char>(totalGreen / totalPixels);
    outputPixel.blue = static_cast<unsigned char>(totalBlue / totalPixels);

    setPixel(image_out, x_dir, y_dir, &outputPixel);
}

/*
void save_image(unsigned char * imagePixel, int width, int height, const char * filename, BMPHeader * headerInfo)
{
    // Write the blurred image to a new file
    std::ofstream outputFile(filename, std::ios::binary);

    if (!outputFile) {
        throw std::invalid_argument("Error: Couldn't create the output BMP image file.");
    }

    // Write BMP header to the output file
    outputFile.write(reinterpret_cast<char*>(&headerInfo), sizeof(BMPHeader));

    std::vector<unsigned char> outputPixels;

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            outputPixels.push_back(imagePixel[i]);
        }
    }

    outputFile.write((const char*)& outputPixels[0], outputPixels.size());
    // Close the output file
    outputFile.close();
}
*/

/*
void create_image(std::vector<unsigned char>& image_copy, int width, int height, const char* inputFilePath, BMPHeader * headerInfo)
{
    std::ifstream inputFile(inputFilePath, std::ios::binary);

    if (!inputFile) 
    {
        throw std::invalid_argument("Error: Couldn't open the input BMP image file.");
    }

    // Read BMP header
    inputFile.read(reinterpret_cast<char*>(&headerInfo), sizeof(BMPHeader));

    // Check if the file is a valid BMP file
    if (headerInfo.fileType != 0x4D42 || headerInfo.bitsPerPixel != 24)
    {
        inputFile.close();
        throw std::invalid_argument("Error: Not a valid BMP file or unsupported color depth.");
    }

    // Move to the beginning of the pixel data
    inputFile.seekg(headerInfo.dataOffset, std::ios::beg);

    // Read BMP pixel data
    std::vector<std::vector<unsigned char>> image(headerInfo.width, std::vector<unsigned char>(headerInfo.height));

    for (int i = 0; i < headerInfo.width; i++)
    {
        for (int j = 0; j < headerInfo.height; j++)
        {
            image_copy.push_back(image[i][j]);
        }
    }

    printf("Image creation done\n");

    inputFile.close();
    return headerInfo;
}
*/

int main(int argc, char * argv[]) 
{   
    if (argc < 4)
    {
        printf("not enough inputs. See ReadMe");
    }
    const char* filename = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    int radius = atoi(argv[4]);

    std::vector<unsigned char> input_image;

    BMPHeader headerInfo; 

    // -------------------------------------------------
    // READ FILE INPUT 
    std::ifstream inputFile(filename, std::ios::binary);

    if (!inputFile)
    {
        throw std::invalid_argument("Error: Couldn't open the input BMP image file.");
    }

    // Read BMP header
    inputFile.read(reinterpret_cast<char*>(&headerInfo), sizeof(BMPHeader));

    // Check if the file is a valid BMP file
    if (headerInfo.fileType != 0x4D42 || headerInfo.bitsPerPixel != 24)
    {
        inputFile.close();
        throw std::invalid_argument("Error: Not a valid BMP file or unsupported color depth.");
    }

    // Move to the beginning of the pixel data
    inputFile.seekg(headerInfo.dataOffset, std::ios::beg);

    // Read BMP pixel data
    std::vector<std::vector<unsigned char>> image(headerInfo.width, std::vector<unsigned char>(headerInfo.height));

    for (int i = 0; i < headerInfo.width; i++)
    {
        for (int j = 0; j < headerInfo.height; j++)
        {
            input_image.push_back(image[i][j]);
        }
    }

    printf("Image creation done\n");

    inputFile.close();    
    
    // --------------------------------------------------------

    unsigned char* pic_in;
    unsigned char* pic_out; 

    int numbytes = sizeof(unsigned char) * width * height * 4;

    cudaMalloc(&pic_in, numbytes);
    cudaMalloc(&pic_out, numbytes);
    // destination, source, size, kind
    cudaMemcpy(pic_in, &input_image[0], numbytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32); // 1024 threads per block
    dim3 numBlocks(width/threadsPerBlock.x, height/threadsPerBlock.y); 

    double start = (1000 * ((double)clock())) / (double)CLOCKS_PER_SEC;
    averageBlur << <numBlocks, threadsPerBlock >> > (pic_in, pic_out, width, height, radius);
    
    cudaDeviceSynchronize();

    unsigned char output_image[numbytes];
    // destination, pitch, source, Woffset, Hoffset, width, height, kind

    cudaMemcpy(output_image, pic_out, numbytes, cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // WRITE OUTPUT 
    // 
    // Write the blurred image to a new file
    printf("writing output\n");
    std::ofstream outputFile("filenameoutput.bmp", std::ios::binary);

    if (!outputFile) {
        throw std::invalid_argument("Error: Couldn't create the output BMP image file.");
    }

    // Write BMP header to the output file
    outputFile.write(reinterpret_cast<char*>(&headerInfo), sizeof(BMPHeader));

        for (int j = 0; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                outputFile.write((const char*)&output_image, width);
            }
        }


    // Close the output file
    outputFile.close();

    // --------------------------------------------------------

    double end = (1000 * ((double)clock())) / (double)CLOCKS_PER_SEC;
    double time_taken = end - start;

    printf("Elapsed time for %d x %d resolution image with coeffient of %d: %f \n", width, height, radius, time_taken);

    return 0;
}


