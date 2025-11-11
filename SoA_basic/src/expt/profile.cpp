#include <ittnotify.h>
#include <iostream>
#include <sstream>
#include <thread>

#include "image/Image.h"
#include "kernel/Kernel.h"
#include "image/reader/STBImageReader.h"
#include "processing/ImageProcessing.h"
#include "kernel/KernelFactory.h"

#ifdef _OPENMP
#include <omp.h>
#endif


int main() {
    constexpr unsigned int imageQuality = 6;
    constexpr unsigned int order = 19;
    int numThreads = 16;

    try {
        __itt_pause();

        // setup num threads
#ifdef _OPENMP
        if (int max_threads = omp_get_max_threads(); numThreads > max_threads) numThreads = max_threads;
        omp_set_num_threads(numThreads);
#else
        numThreads = 1;
#endif
        std::cout << "Using " << numThreads << " thread(s)." << std::endl << std::endl;

        // setup image reader
        STBImageReader imageReader{};
        std::stringstream fullPathStream;

        const std::string imageName = std::to_string(imageQuality) + "K-1";

        // load img
        fullPathStream << IMAGES_INPUT_DIRPATH << imageName << ".jpg";
        const auto img = imageReader.loadRGBImage(fullPathStream.str());
        std::cout << "Image " << imageName << " (" << img->getWidth() << "x" << img->getHeight() <<
            ") loaded from: " << fullPathStream.str() << std::endl;
        fullPathStream.str(std::string());

        // enlargement
        const auto extendedImage = ImageProcessing::extendEdge(*img, (order - 1) / 2);
        std::cout << "Image "  << imageName << " enlarged to " <<
            extendedImage->getWidth() << "x" << extendedImage->getHeight() << std::endl;

        // create kernel
        const auto kernel = KernelFactory::createBoxBlurKernel(order);
        std::cout << "Kernel \"" << kernel->getName() << "\" " << kernel->getOrder() << "x" << kernel->getOrder() <<
            " created." << std::endl;

        // transform
        __itt_resume();
        const auto outputImage = ImageProcessing::convolution(*extendedImage, *kernel);
        __itt_pause();

        // save
        fullPathStream << IMAGES_OUTPUT_DIRPATH << imageName << "_" << kernel->getName() << kernel->getOrder() << ".jpg";
        imageReader.saveJPGImage(*outputImage, fullPathStream.str());
        std::cout << "Image " << outputImage->getWidth() << "x" << outputImage->getHeight() <<
            " saved at: " << fullPathStream.str() << std::endl << std::endl;
        fullPathStream.str(std::string());

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
