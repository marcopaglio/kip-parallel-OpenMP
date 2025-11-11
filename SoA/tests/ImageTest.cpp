#include "gtest/gtest.h"
#include "image/Image.h"


TEST(ImageTest, testConstructor) {
    constexpr unsigned int height = 3;
    constexpr unsigned int width = 5;
    const std::vector<uint8_t> reds = {120, 23, 44, 123, 1,
                            1, 19, 67, 88, 81,
                            43, 100, 215, 10, 90};
    const std::vector<uint8_t> greens = {0, 58, 30, 15, 12,
                                17, 89, 12, 137, 3,
                                38, 10, 35, 4, 36};
    const std::vector<uint8_t> blues = {130, 135, 20, 15, 68,
                                225, 139, 29, 213, 64,
                                106, 0, 120, 65, 217};

    Image image(width, height, reds, greens, blues);

    EXPECT_EQ(image.getWidth(), width);
    EXPECT_EQ(image.getHeight(), height);
    ASSERT_EQ(image.getReds().size(), width * height);
    ASSERT_EQ(image.getGreens().size(), width * height);
    ASSERT_EQ(image.getBlues().size(), width * height);
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            EXPECT_EQ(image.getReds()[y * width + x], reds[y * width + x]);
            EXPECT_EQ(image.getGreens()[y * width + x], greens[y * width + x]);
            EXPECT_EQ(image.getBlues()[y * width + x], blues[y * width + x]);
        }
    }
}