#ifndef STBIMAGEREADER_H
#define STBIMAGEREADER_H
#include <filesystem>

#include "image/Image.h"
#include "ImageReader.h"


/**
 * A final concrete implementation of the @ref ImageReader interface for reading and writing images using STB library.
 */
class STBImageReader final : public ImageReader {
public:
    /**
     * Default constructor.
     */
    STBImageReader();

    /**
     * Default destructor.
     */
    ~STBImageReader() override;


    /**
     * Loads an RGB image from the specified file path using STB library.
     *
     * @param filePath The full or relative file path to the image to load.
     * @return A unique pointer to the Image object containing the loaded image data.
     * @throw std::runtime_error If the image fails to load.
     */
    std::unique_ptr<Image> loadRGBImage(const std::filesystem::path& filePath) override;

    /**
     * Saves an image in JPEG format to the specified file path using STB library.
     *
     * @param img The Image object to save.
     * @param filePath The full or relative file path where the JPEG image will be saved.
     * @throw std::runtime_error If the image fails to save.
     */
    void saveJPGImage(const Image &img, const std::filesystem::path &filePath) override;
};



#endif //STBIMAGEREADER_H
