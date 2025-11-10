#include "ImageProcessing.h"

// #ifdef _OPENMP
// #include <omp.h>
// #endif

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

// #pragma omp parallel collapse(2) schedule(guided)     > prestazioni simili poiché collapse(2) rimuove la località spaziale
#pragma omp parallel for schedule(dynamic) default(none) \
    shared(pixels, originalData, outputHeight) \
    firstprivate(outputWidth, order, kernelWeights)
//{
    /**
     * Shared vs Firstprivate
     * - Variabili piccole e super usate nei calcoli → provare firstprivate può dare un micro-vantaggio.
     * - Variabili più grandi o usate meno intensivamente → shared è preferibile, perché evita costi di copia.
     *
     * A tal proposito:
     * - pixels: scritto una volta per thread e alla fine → shared perfetto
     * - originalData: read-only, ma molto grande per cui non vale l'overhead della copia → shared
     * - outputHeight: read-only, ma una volta per thread → shared
     * - outputWidth: read-only, ma SENZA COLLAPSE diverse volte per thread → firstprivate
     *                           ma CON COLLAPSE una volta per thread → shared
     * - order: read-only, ma tante volte → firstprivate
     * - kernelWeights: come order, ma più overhead per la copia → firstprivate (sperimentalmente conviene a shared)
     *
     */
// manually work division (in horizontal stripes) is a little better than static scheduler, but worse than dynamic one
//     unsigned int lowerBound;
//     unsigned int upperBound;
// #ifdef _OPENMP
//     int thread_id = omp_get_thread_num();
//     int nthreads = omp_get_num_threads();
//
//     lowerBound = outputHeight / nthreads * thread_id;
//     upperBound = thread_id == nthreads - 1 ? outputHeight : outputHeight / nthreads * (thread_id + 1);
//
//     printf("Thread n.%d has lower=%d and upper=%d\n", thread_id, lowerBound, upperBound);
// #else
//     lowerBound = 0;
//     upperBound = outputHeight;
// #endif
    for (unsigned int y = 0; y < outputHeight; y++) {
        for (unsigned int x = 0; x < outputWidth; x++) {
            float channelRed = 0;
            float channelGreen = 0;
            float channelBlue = 0;

// #pragma omp simd reduction(+:channelRed, channelGreen, channelBlue)  > Prestazioni identiche, probabilmente in Release viene già vettorizzato
            for (unsigned int j = 0; j < order; j++) {
                for (unsigned int i = 0; i < order; i++) {
                    Pixel originalPixel = originalData[y + j][x + i];
                    const float kernelWeight = kernelWeights[j * order + i];
                    channelRed += static_cast<float>(originalPixel.getR()) * kernelWeight;
                    channelGreen += static_cast<float>(originalPixel.getG()) * kernelWeight;
                    channelBlue += static_cast<float>(originalPixel.getB()) * kernelWeight;
                }
            }
            pixels[y][x] = Pixel(getChannelAsUint8(channelRed),
                getChannelAsUint8(channelGreen), getChannelAsUint8(channelBlue));
        }
    }
//} // end omp parallel

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