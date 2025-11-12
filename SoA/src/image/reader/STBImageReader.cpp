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
    std::vector<uint8_t> reds (width * height);
    std::vector<uint8_t> greens (width * height);
    std::vector<uint8_t> blues (width * height);
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            const unsigned int pos = y * width + x;
            const unsigned int idx = (y * width + x) * RGB_CHANNELS;
            reds[pos] = imgData[idx];
            greens[pos] = imgData[idx + 1];
            blues[pos] = imgData[idx + 2];
        }
    }

    stbi_image_free(imgData);
    return std::make_unique<Image>(width, height, reds, greens, blues);
}

void STBImageReader::saveJPGImage(const Image &img, const std::filesystem::path &filePath) {
    const unsigned int width = img.getWidth();
    const unsigned int height = img.getHeight();

    std::vector<uint8_t> flatData(width * height * RGB_CHANNELS);

    // retrieve data
    const auto reds = img.getReds();
    const auto greens = img.getGreens();
    const auto blues = img.getBlues();
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            const unsigned int idx = (y * width + x) * RGB_CHANNELS;
            const unsigned int pos = y * width + x;
            flatData[idx] = reds[pos];
            flatData[idx + 1] = greens[pos];
            flatData[idx + 2] = blues[pos];
        }
    }

    // save
    if (!stbi_write_jpg(filePath.generic_string().c_str(), static_cast<int>(width), static_cast<int>(height),
        RGB_CHANNELS, flatData.data(), JPG_QUALITY)) {
        throw std::runtime_error("Image saving fails.");
    }
}