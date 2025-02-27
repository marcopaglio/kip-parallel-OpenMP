#ifndef STBIMAGEREADER_H
#define STBIMAGEREADER_H
#include <filesystem>

#include "Image.h"
#include "ImageReader.h"


class STBImageReader final : public ImageReader {
public:
    STBImageReader();
    ~STBImageReader() override;

    /**
     * Load an image as an RGB Image.
     *
     * @param filePath the full or relative path with the name of the image to load.
     * @return a smart pointer to an RGB Image object.
     * @throw runtime_error if loading fails.
     */
    std::unique_ptr<Image> loadRGBImage(const std::filesystem::path& filePath) override;

    /**
     * Save an Image as JPG in local disk.
     *
     * @param img the Image to save.
     * @param filePath the full or relative path with the name in which to save the image.
     * @throw runtime_error if saving fails.
     */
    void saveJPGImage(const Image &img, const std::filesystem::path &filePath) override;
};



#endif //STBIMAGEREADER_H
