#include <iostream>
#include <sstream>
#include <thread>

#include "image/Image.h"
#include "kernel/Kernel.h"
#include "image/reader/STBImageReader.h"
#include "processing/ImageProcessing.h"
#include "kernel/KernelFactory.h"


int main() {
    constexpr unsigned int imageQuality = 6;
    constexpr unsigned int order = 19;

    try {
        // setup image reader
        STBImageReader imageReader{};
        std::stringstream fullPathStream;

        const std::string imageName = std::to_string(imageQuality) + "K-1";

        // load img
        fullPathStream << PROJECT_SOURCE_DIR << "/imgs/input/" << imageName << ".jpg";
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
        const auto outputImage = ImageProcessing::convolution(*extendedImage, *kernel);

        // save
        fullPathStream << PROJECT_SOURCE_DIR << "/imgs/output/" << imageName <<
            "_" << kernel->getName() << kernel->getOrder() << "_profile.jpg";
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
