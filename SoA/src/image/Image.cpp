#include "Image.h"

Image::Image(const unsigned int w, const unsigned int h, const std::vector<uint8_t>& reds,
    const std::vector<uint8_t>& greens, const std::vector<uint8_t>& blues):
    width(w), height(h), reds(reds), greens(greens), blues(blues) {}

Image::~Image() = default;

unsigned int Image::getWidth() const {
    return width;
}

unsigned int Image::getHeight() const {
    return height;
}

std::vector<uint8_t> Image::getReds() const {
    return reds;
}

std::vector<uint8_t> Image::getGreens() const {
    return greens;
}

std::vector<uint8_t> Image::getBlues() const {
    return blues;
}