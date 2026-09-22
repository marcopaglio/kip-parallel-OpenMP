#include "ImageProcessing.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define MIN_VALUE 0
#define MAX_VALUE 255

uint8_t getChannelAsUint8(const float channel) {
    if (channel < MIN_VALUE)
        return MIN_VALUE;
    if (channel > MAX_VALUE)
        return MAX_VALUE;
    return static_cast<uint8_t>(channel);
}

std::unique_ptr<Image> ImageProcessing::convolution(const Image &image, const Kernel &kernel) {
    const unsigned int order = kernel.getOrder();
    const auto kernelWeights = kernel.getWeights();

    const auto originalData = image.getData();
    const unsigned int outputHeight = image.getHeight() - (order - 1);
    const unsigned int outputWidth = image.getWidth() - (order - 1);

    std::vector pixels(outputHeight, std::vector<Pixel>(outputWidth));

    constexpr unsigned int TILE_X = 1024;

#pragma omp parallel for schedule(dynamic) default(none) \
shared(pixels, originalData, outputHeight, TILE_X) \
firstprivate(outputWidth, order, kernelWeights)
    for (unsigned int y = 0; y < outputHeight; y++) {
        float channelReds[TILE_X];
        float channelGreens[TILE_X];
        float channelBlues[TILE_X];

        for (unsigned int xBegin = 0; xBegin < outputWidth; xBegin += TILE_X) {
            const unsigned int tileLength = std::min(TILE_X, outputWidth - xBegin);

            // 1. Azzera canali RGB del tile
            for (unsigned int t = 0; t < tileLength; t++) {
                channelReds[t] = 0.0f;
                channelGreens[t] = 0.0f;
                channelBlues[t] = 0.0f;
            }

            // 2. Applica convoluzione al tile
            for (unsigned int j = 0; j < order; j++) {
                const unsigned int posBaseY = y + j;
                const unsigned int kwBase = j * order;

                for (unsigned int i = 0; i < order; i++) {
                    const unsigned int posBaseX = xBegin + i;
                    const float kernelWeight = kernelWeights[kwBase + i];

                    for (unsigned int  t = 0; t < tileLength; t++) {
                        const unsigned int posTileX = posBaseX + t;

                        Pixel originalPixel = originalData[posBaseY][posTileX];
                        channelReds[t] += static_cast<float>(originalPixel.getR()) * kernelWeight;
                        channelGreens[t] += static_cast<float>(originalPixel.getG()) * kernelWeight;
                        channelBlues[t] += static_cast<float>(originalPixel.getB()) * kernelWeight;
                    }
                }
            }

            // 3. Copia i canali RGB del tile
            for (unsigned int t = 0; t < tileLength; t++)
                pixels[y][xBegin + t] = Pixel(getChannelAsUint8(channelReds[t]),
                    getChannelAsUint8(channelGreens[t]), getChannelAsUint8(channelBlues[t]));
        }
    }

    return std::make_unique<Image>(outputWidth, outputHeight, pixels);
}


std::unique_ptr<Image> ImageProcessing::extendEdge(const Image &image, const unsigned int padding) {
    const auto originalData = image.getData();
    const unsigned int height = image.getHeight();
    const unsigned int width = image.getWidth();

    const unsigned int extendedHeight = height + 2 * padding;
    const unsigned int extendedWidth = width + 2 * padding;

    std::vector pixels(extendedHeight, std::vector<Pixel>(extendedWidth));
    // copy image main data
    for (unsigned int j = 0; j < height; j++) {
        for (unsigned int i = 0; i < width; i++) {
            pixels[j + padding][i + padding] = originalData[j][i];
        }
    }

    // fill left internal new columns
    for (unsigned int j = 0; j < height; j++) {
        for (unsigned int i = 0; i < padding; i++) {
            pixels[j + padding][i] = originalData[j][0];
        }
    }

    // fill right internal new columns
    for (unsigned int j = 0; j < height; j++) {
        for (unsigned int i = 0; i < padding; i++) {
            pixels[j + padding][padding + width + i] = originalData[j][width - 1];
        }
    }

    // fill top internal new rows
    for (unsigned int j = 0; j < padding; j++) {
        for (unsigned int i = 0; i < width; i++) {
            pixels[j][padding + i] = originalData[0][i];
        }
    }

    // fill bottom internal new rows
    for (unsigned int j = 0; j < padding; j++) {
        for (unsigned int i = 0; i < width; i++) {
            pixels[padding + height + j][padding + i] = originalData[height - 1][i];
        }
    }

    // fill corners
    for (unsigned int j = 0; j < padding; j++) {
        for (unsigned int i = 0; i < padding; i++) {
            pixels[j][i] = originalData[0][0];                                                              // top-left
            pixels[extendedHeight - 1 - j][i] = originalData[height - 1][0];                                // bottom-left
            pixels[j][extendedWidth - 1 - i] = originalData[0][width - 1];                                  // top-right
            pixels[extendedHeight - 1 - j][extendedWidth - 1 - i] = originalData[height - 1][width - 1];    // bottom-right
        }
    }

    return std::make_unique<Image>(extendedWidth, extendedHeight, pixels);
}