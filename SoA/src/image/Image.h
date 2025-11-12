#ifndef IMAGE_H
#define IMAGE_H
#include <vector>


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
     * @param reds A vector containing red component's values for the image.
     * @param greens A vector containing green component's values for the image.
     * @param blues A vector containing blue component's values for the image.
     */
    Image(unsigned int w, unsigned int h, const std::vector<uint8_t>& reds,
        const std::vector<uint8_t>& greens, const std::vector<uint8_t>& blues);

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
     * Retrieves the red components of the image.
     *
     * @return The image's red channel values as a vector of 8-bit unsigned integer.
     */
    [[nodiscard]] std::vector<uint8_t> getReds() const;

    /**
     * Retrieves the green components of the image.
     *
     * @return The image's green channel values as a vector of 8-bit unsigned integer.
     */
    [[nodiscard]] std::vector<uint8_t> getGreens() const;

    /**
     * Retrieves the blue components of the image.
     *
     * @return The image's blue channel values as a vector of 8-bit unsigned integer.
     */
    [[nodiscard]] std::vector<uint8_t> getBlues() const;

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
     * Stores the red components for the whole image as a vector.
     *
     * Each red value is stored as an 8-bit unsigned integer, allowing values
     * between 0 and 255 to represent the intensity of the color channel.
     */
    std::vector<uint8_t> reds;

    /**
     * Stores the green components for the whole image as a vector.
     *
     * Each green value is stored as an 8-bit unsigned integer, allowing values
     * between 0 and 255 to represent the intensity of the color channel.
     */
    std::vector<uint8_t> greens;

    /**
     * Stores the blue components for the whole image as a vector.
     *
     * Each blue value is stored as an 8-bit unsigned integer, allowing values
     * between 0 and 255 to represent the intensity of the color channel.
     */
    std::vector<uint8_t> blues;
};



#endif //IMAGE_H
