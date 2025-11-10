#ifndef IMAGE_H
#define IMAGE_H
#include <vector>

#include "Pixel.h"


/**
 * Represents an image composed of pixel data in a two-dimensional structure.
 * The image is defined by its width, height, and pixel information.
 *
 * This class is immutable once constructed.
 */
class Image {
public:
    /**
     * Constructs an Image object with the specified width, height, and pixel data.
     *
     * @param w The width of the image in pixels.
     * @param h The height of the image in pixels.
     * @param data A 2D vector containing pixel information for the image. The outer vector corresponds
     *             to rows, and the inner vectors represent columns of pixels in the image.
     */
    Image(unsigned int w, unsigned int h, const std::vector<std::vector<Pixel>>& data);

    /**
     * Default destructor.
     */
    ~Image();

    /**
     * Retrieves the width of the object.
     *
     * @return The width of the object as an integer.
     */
    [[nodiscard]] unsigned int getWidth() const;

    /**
     * Retrieves the height of the object.
     *
     * @return The height value as an integer.
     */
    [[nodiscard]] unsigned int getHeight() const;

    /**
     * Retrieves the pixel data of the image.
     *
     * @return A 2D vector containing the image's pixel data. Each inner vector represents a row of
     *         pixels, while its elements correspond to the columns of the image.
     */
    [[nodiscard]] std::vector<std::vector<Pixel>> getData() const;

private:
    /**
     * Represent the width of the image in pixels.
     *
     * The width determines the number of columns in the 2D pixel data of the image.
     */
    unsigned int width;

    /**
     * Represents the height of the image in pixels.
     *
     * The height determines the number of rows in the 2D pixel data for the image.
     */
    unsigned int height;

    /**
     * Stores the pixel data for the image as a 2D vector of Pixel objects.
     *
     * Each outer vector represents a row of pixels, while each inner vector contains
     * the pixels for a specific column within that row.
     */
    std::vector<std::vector<Pixel>> data;
};



#endif //IMAGE_H
