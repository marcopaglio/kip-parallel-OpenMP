#include "stb_image.h"

#include "Workload.h"

#define RGB_CHANNELS 3

std::unique_ptr<Image> workload::loadExpandedRGBImage(const std::filesystem::path &filePath, const unsigned int numLoads) {
    int width, height, channels;

    unsigned char* imgData = stbi_load(filePath.generic_string().c_str(), &width, &height, &channels, RGB_CHANNELS);
    if (!imgData) {
        throw std::runtime_error("Image loading fails.");
    }

    // expand only in high
    const unsigned int expandedHeight = numLoads * height;

    // conversion
    std::vector pixels(expandedHeight, std::vector<Pixel>(width));
    for (int w = 0; w < numLoads; w++) {
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                const unsigned int idx = (y * width + x) * RGB_CHANNELS;
                pixels[w * height + y][x] = Pixel(imgData[idx], imgData[idx + 1], imgData[idx + 2]);
            }
        }
    }

    stbi_image_free(imgData);
    return std::make_unique<Image>(width, expandedHeight, pixels);
}
