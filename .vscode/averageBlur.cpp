#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>

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

Pixel averageColor(const std::vector<std::vector<Pixel>>& image, int x, int y, int radius) {
    int totalPixels = 0;
    int totalRed = 0, totalGreen = 0, totalBlue = 0;

    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            int newX = x + i;
            int newY = y + j;

            if (newX >= 0 && newX < image.size() && newY >= 0 && newY < image[0].size()) {
                totalRed += image[newX][newY].red;
                totalGreen += image[newX][newY].green;
                totalBlue += image[newX][newY].blue;
                ++totalPixels;
            }
        }
    }

    Pixel averagePixel;
    averagePixel.red = static_cast<unsigned char>(totalRed / totalPixels);
    averagePixel.green = static_cast<unsigned char>(totalGreen / totalPixels);
    averagePixel.blue = static_cast<unsigned char>(totalBlue / totalPixels);

    return averagePixel;
}

void applyMediumBlur(std::vector<std::vector<Pixel>>& image, int radius) {
    //const int radius = 3;  // Radius of the medium blur filter

    std::vector<std::vector<Pixel>> blurredImage = image;

    for (int x = 0; x < image.size(); ++x) {
        for (int y = 0; y < image[x].size(); ++y) {
            blurredImage[x][y] = averageColor(image, x, y, radius);
        }
    }

    image = blurredImage;
}

int main() {
    std::string inputFilePath, outputFilePath;
    int radius;

    std::cout << "Enter the input BMP image file path: ";
    std::cin >> inputFilePath;

    std::cout << "Enter radius: ";
    std::cin >> radius;

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
    std::vector<std::vector<Pixel>> image(bmpHeader.width, std::vector<Pixel>(bmpHeader.height));

    // BMP data is stored upside down, so we read the rows in reverse order
    for (int y = bmpHeader.height - 1; y >= 0; --y) {
        for (int x = 0; x < bmpHeader.width; ++x) {
            inputFile.read(reinterpret_cast<char*>(&image[x][y]), sizeof(Pixel));
        }

        // Skip any padding (if present) to align rows to multiples of 4 bytes
        inputFile.seekg((4 - (bmpHeader.width * 3) % 4) % 4, std::ios::cur);
    }

    // Close the input file
    inputFile.close();

    // Apply medium blur to the image
    applyMediumBlur(image, radius);;

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
            outputFile.write(reinterpret_cast<char*>(&image[x][y]), sizeof(Pixel));
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