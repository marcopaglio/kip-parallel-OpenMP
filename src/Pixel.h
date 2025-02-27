#ifndef PIXEL_H
#define PIXEL_H
#include <cstdint>


class Pixel final {
public:
    explicit Pixel(uint8_t r=0, uint8_t g=0, uint8_t b=0);
    ~Pixel();

    [[nodiscard]] uint8_t getR() const;
    [[nodiscard]] uint8_t getG() const;
    [[nodiscard]] uint8_t getB() const;

private:
    uint8_t r, g, b;
};



#endif //PIXEL_H
