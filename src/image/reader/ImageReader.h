#ifndef IMAGEREADER_H
#define IMAGEREADER_H
#include <filesystem>

#include "image/Image.h"


/**
 * Represents an interface for reading and writing image files.
 */
class ImageReader {
public:
    /**
     * Default constructor.
     */
    ImageReader();

    /**
     * Default destructor.
     *
     * It is declared as virtual to ensure that the proper destructor is called for derived classes.
     */
    virtual ~ImageReader();


    /**
     * Loads an RGB image from the specified file path.
     *
     * @param filePath The full or relative file path to the image to load.
     * @return A unique pointer to the Image object containing the loaded image data.
     * @throw std::runtime_error If the image fails to load.
     */
    virtual std::unique_ptr<Image> loadRGBImage(const std::filesystem::path& filePath) = 0;

    /**
     * Saves an image in JPEG format to the specified file path.
     *
     * @param img The Image object to save.
     * @param filePath The full or relative file path where the JPEG image will be saved.
     * @throw std::runtime_error If the image fails to save.
     */
    virtual void saveJPGImage(const Image& img, const std::filesystem::path& filePath) = 0;
};



#endif //IMAGEREADER_H
