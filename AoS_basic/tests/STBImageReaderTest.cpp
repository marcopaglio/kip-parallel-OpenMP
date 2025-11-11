#include <gtest/gtest.h>
#include <filesystem>
#include "image/reader/STBImageReader.h"

class STBImageReaderTest : public ::testing::Test {
protected:
    STBImageReader* imageReader = nullptr;

    void SetUp() override {
        imageReader = new STBImageReader();
    }

    void TearDown() override {
        delete imageReader;
        imageReader = nullptr;
    }
};


TEST_F(STBImageReaderTest, testLoadRGBImageWhenImageExists) {
    constexpr unsigned int width = 5;
    constexpr unsigned int height = 3;
    const Pixel rgb00(120, 1, 131);
    const Pixel rgb01(22, 58, 136);
    const Pixel rgb02(45, 31, 22);
    const Pixel rgb03(123, 15, 15);
    const Pixel rgb04(1, 12, 68);
    const std::vector row0 = {rgb00, rgb01, rgb02, rgb03, rgb04};
    const Pixel rgb10(1, 17, 226);
    const Pixel rgb11(18, 88, 140);
    const Pixel rgb12(66, 12, 28);
    const Pixel rgb13(89, 137, 213);
    const Pixel rgb14(82, 4, 64);
    const std::vector row1 = {rgb10, rgb11, rgb12, rgb13, rgb14};
    const Pixel rgb20(42, 36, 106);
    const Pixel rgb21(100, 10, 1);
    const Pixel rgb22(215, 36, 118);
    const Pixel rgb23(10, 4, 64);
    const Pixel rgb24(90, 36, 217);
    const std::vector row2 = {rgb20, rgb21, rgb22, rgb23, rgb24};
    const std::vector pixels = {row0, row1, row2};

    std::stringstream inputFilePathStream;
    inputFilePathStream << TEST_IMAGES_INPUT_DIRPATH << "testImage.jpg";
    const std::string inputFilePath = inputFilePathStream.str();

    const auto img = imageReader->loadRGBImage(inputFilePath);

    ASSERT_NE(img, nullptr);
    EXPECT_EQ(img->getHeight(), height);
    EXPECT_EQ(img->getWidth(), width);
    ASSERT_EQ(img->getData().size(), height);
    ASSERT_EQ(img->getData()[0].size(), width);
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            EXPECT_EQ(img->getData()[y][x].getR(), pixels[y][x].getR());
            EXPECT_EQ(img->getData()[y][x].getG(), pixels[y][x].getG());
            EXPECT_EQ(img->getData()[y][x].getB(), pixels[y][x].getB());
        }
    }
}

TEST_F(STBImageReaderTest, testLoadRGBImageWhenImageDoesntExist) {
    const std::string inputFilePath = "this/path/doesnt/exist/testImage.jpg";
    std::filesystem::remove_all(inputFilePath);

    EXPECT_THROW(imageReader->loadRGBImage(inputFilePath), std::runtime_error);
}


TEST_F(STBImageReaderTest, testSaveJPGImageWhenPathExists) {
    constexpr unsigned int height = 3;
    constexpr unsigned int width = 5;
    const std::vector aRow(width, Pixel());
    const std::vector somePixels(height, aRow);
    const Image testImage(width, height, somePixels);

    std::stringstream outputFilePathStream;
    outputFilePathStream << TEST_IMAGES_OUTPUT_DIRPATH << "testImage.jpg";
    const std::string outputFilePath = outputFilePathStream.str();

    remove(outputFilePath.c_str());
    ASSERT_FALSE(std::filesystem::exists(outputFilePath));

    imageReader->saveJPGImage(testImage, outputFilePath);

    EXPECT_TRUE(std::filesystem::exists(outputFilePath));
}

TEST_F(STBImageReaderTest, testSaveJPGImageWhenPathDoesntExist) {
    constexpr unsigned int height = 3;
    constexpr unsigned int width = 5;
    const std::vector aRow(width, Pixel());
    const std::vector somePixels(height, aRow);
    const Image testImage(width, height, somePixels);

    const std::string outputFilePath = "this/path/doesnt/exist/testImage.jpg";
    std::filesystem::remove_all(outputFilePath);

    EXPECT_THROW(imageReader->saveJPGImage(testImage, outputFilePath), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(outputFilePath));
}