#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H
#include <memory>

#include "Image.h"
#include "Kernel.h"


namespace ImageProcessing {
    std::unique_ptr<Image> convolution(const Image& image, const Kernel& kernel);
    std::unique_ptr<Image> extendEdge(const Image &image, unsigned int padding);
}



#endif //IMAGEPROCESSING_H
