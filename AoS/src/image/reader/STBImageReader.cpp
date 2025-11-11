#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "STBImageReader.h"

#define RGB_CHANNELS 3
#define JPG_QUALITY 100

STBImageReader::STBImageReader() = default;

STBImageReader::~STBImageReader() = default;

std::unique_ptr<Image> STBImageReader::loadRGBImage(const std::filesystem::path &filePath) {
    int width, height, channels;

    unsigned char* imgData = stbi_load(filePath.generic_string().c_str(), &width, &height, &channels, RGB_CHANNELS);
    if (!imgData) {
        throw std::runtime_error("Image loading fails.");
    }

    // conversion
    std::vector pixels(height, std::vector<Pixel>(width));
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            const unsigned int idx = (y * width + x) * RGB_CHANNELS;
            pixels[y][x] = Pixel(imgData[idx], imgData[idx + 1], imgData[idx + 2]);
        }
    }

    stbi_image_free(imgData);
    return std::make_unique<Image>(width, height, pixels);
}

void STBImageReader::saveJPGImage(const Image &img, const std::filesystem::path &filePath) {
    const unsigned int width = img.getWidth();
    const unsigned int height = img.getHeight();

    std::vector<uint8_t> flatData(width * height * RGB_CHANNELS);

    // retrieve data
    const auto pixels = img.getData();
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            const unsigned int idx = (y * width + x) * RGB_CHANNELS;
            Pixel pixel = pixels[y][x];
            flatData[idx] = pixel.getR();
            flatData[idx + 1] = pixel.getG();
            flatData[idx + 2] = pixel.getB();
        }
    }

    // save
    if (!stbi_write_jpg(filePath.generic_string().c_str(), static_cast<int>(width), static_cast<int>(height),
        RGB_CHANNELS, flatData.data(), JPG_QUALITY)) {
        throw std::runtime_error("Image saving fails.");
    }
}