#ifndef IMAGE_H
#define IMAGE_H
#include <vector>

#include "Pixel.h"


class Image {
public:
    Image(unsigned int w, unsigned int h, const std::vector<std::vector<Pixel>>& data);
    ~Image();

    [[nodiscard]] unsigned int getWidth() const;
    [[nodiscard]] unsigned int getHeight() const;
    [[nodiscard]] std::vector<std::vector<Pixel>> getData() const;

private:
    unsigned int width, height;
    std::vector<std::vector<Pixel>> data;
};



#endif //IMAGE_H
