#include <iostream>
#include <sstream>

#include "HighResolutionTimer.h"
#include "Image.h"
#include "Kernel.h"
#include "STBImageReader.h"
#include "ImageProcessing.h"
#include "KernelFactory.h"
#include "SteadyTimer.h"
#include "Timer.h"

int main() {
    std::unique_ptr<Timer> timer;
    if constexpr (std::chrono::high_resolution_clock::is_steady)
        timer = std::make_unique<HighResolutionTimer>();
    else
        timer = std::make_unique<SteadyTimer>();

    try {
        const std::string imageName = "tuscany";
        std::stringstream fullPathStream;
        STBImageReader imageReader{};

        // load img
        fullPathStream << PROJECT_SOURCE_DIR << "/imgs/input/" << imageName << ".jpg";
        const auto img = imageReader.loadRGBImage(fullPathStream.str());
        std::cout << "Image " << img->getWidth() << "x" << img->getHeight() <<
            " loaded from: " << fullPathStream.str() << std::endl;
        fullPathStream.str(std::string());

        // create kernel
        constexpr unsigned int order = 7;
        //const auto kernel = KernelFactory::createBlurKernel(order);
        const auto kernel = KernelFactory::createEdgeDetectionKernel(order);

        // transform
        const std::chrono::duration<double> wall_clock_time_start = timer->now();
        const std::unique_ptr<Image> outputImage = ImageProcessing::convolution(
                *ImageProcessing::extendEdge(*img, (order - 1) / 2),
                *kernel);
        const std::chrono::duration<double> wall_clock_time_end = timer->now();
        const std::chrono::duration<double> wall_clock_time_duration = wall_clock_time_end - wall_clock_time_start;
        std::cout << "Finished in " << wall_clock_time_duration.count() << " seconds [Wall Clock]" << std::endl;

        // save
        fullPathStream << PROJECT_SOURCE_DIR << "/imgs/output/" << imageName <<
            "_" << kernel->getName() << kernel->getOrder() << ".jpg";
        imageReader.saveJPGImage(*outputImage, fullPathStream.str());
        std::cout << "Image " << outputImage->getWidth() << "x" << outputImage->getHeight() <<
            " saved at: " << fullPathStream.str() << std::endl;
        fullPathStream.str(std::string());

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}