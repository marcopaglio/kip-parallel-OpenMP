#ifndef IMAGEREADER_H
#define IMAGEREADER_H
#include <filesystem>

#include "Image.h"


class ImageReader {
public:
    ImageReader();
    virtual ~ImageReader();

    virtual std::unique_ptr<Image> loadRGBImage(const std::filesystem::path& filePath) = 0;
    virtual void saveJPGImage(const Image& img, const std::filesystem::path& filePath) = 0;
};



#endif //IMAGEREADER_H
