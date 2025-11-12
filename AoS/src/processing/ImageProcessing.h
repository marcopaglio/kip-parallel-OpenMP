#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H
#include <memory>

#include "image/Image.h"
#include "kernel/Kernel.h"


/**
 * Namespace for operations involving images.
 */
namespace ImageProcessing {
    /**
     * Applies a convolution operation on the given image using the specified kernel.
     *
     * This operation reduces the size of the transformed image relative to the input image
     * by a number of pixels equal to the kernel order.
     * To obtain an image without cropping, use @ref extendEdge before performing this operation.
     *
     * @param image The input image on which the convolution operation will be performed.
     * @param kernel The kernel used for the convolution.
     * @return A unique pointer to a new Image object containing the result of the convolution.
     */
    std::unique_ptr<Image> convolution(const Image& image, const Kernel& kernel);

    /**
     * Extends the edges of the given image by padding a specified number of pixels around it.
     * The edge pixels are replicated outward to fill the padding region.
     *
     * @param image The original image to be padded.
     * @param padding The number of pixels to add around each edge of the input image.
     * @return A unique pointer to a new Image object with extended edges.
     */
    std::unique_ptr<Image> extendEdge(const Image &image, unsigned int padding);
}



#endif //IMAGEPROCESSING_H
