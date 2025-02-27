#include "Pixel.h"

Pixel::Pixel(const uint8_t r, const uint8_t g, const uint8_t b): r(r), g(g), b(b) {}

Pixel::~Pixel() = default;

uint8_t Pixel::getR() const {
    return r;
}

uint8_t Pixel::getG() const {
    return g;
}

uint8_t Pixel::getB() const {
    return b;
}