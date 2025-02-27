#include <gtest/gtest.h>
#include "../src/Pixel.h"


TEST(PixelTest, testConstructor) {
    constexpr uint8_t r = 13;
    constexpr uint8_t g  = 28;
    constexpr uint8_t b = 234;

    const Pixel pixel(r, g, b);

    EXPECT_EQ(pixel.getR(), r);
    EXPECT_EQ(pixel.getG(), g);
    EXPECT_EQ(pixel.getB(), b);
}

TEST(PixelTest, testDefaultConstructor) {
    const Pixel defaultPixel;

    EXPECT_EQ(defaultPixel.getR(), 0);
    EXPECT_EQ(defaultPixel.getG(), 0);
    EXPECT_EQ(defaultPixel.getB(), 0);
}