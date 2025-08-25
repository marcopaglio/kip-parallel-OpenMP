#ifndef PIXEL_H
#define PIXEL_H
#include <cstdint>


/**
 * A class representing a single RGB pixel.
 *
 * This class is immutable once constructed.
 */
class Pixel final {
public:
    /**
     * Constructs a Pixel object with the specified red, green, and blue color components.
     *
     * @param r The red color component, represented as an 8-bit unsigned integer (range: 0-255).
     * @param g The green color component, represented as an 8-bit unsigned integer (range: 0-255).
     * @param b The blue color component, represented as an 8-bit unsigned integer (range: 0-255).
     */
    explicit Pixel(uint8_t r=0, uint8_t g=0, uint8_t b=0);

    /**
     * Default destructor.
     */
    ~Pixel();

    /**
     * Retrieves the red component of the pixel.
     *
     * @return The red color value of the pixel as an 8-bit unsigned integer.
     */
    [[nodiscard]] uint8_t getR() const;

    /**
     * Retrieves the green color component of the Pixel object.
     *
     * @return The value of the green component as an unsigned 8-bit integer.
     */
    [[nodiscard]] uint8_t getG() const;

    /**
     * Retrieves the blue color value of the Pixel.
     *
     * @return The blue component of the Pixel as an 8-bit unsigned integer.
     */
    [[nodiscard]] uint8_t getB() const;

private:
    /**
     * Represents the red, green, and blue color components of a pixel.
     *
     * Each color component is stored as an 8-bit unsigned integer, allowing values
     * between 0 and 255 to represent the intensity of the respective color channel.
     */
    uint8_t r, g, b;
};



#endif //PIXEL_H
