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
    const std::vector reds = {120, 22, 45, 123, 1,
                                1, 18, 66, 89, 82,
                                42, 100, 215, 10, 90};
    const std::vector greens = {1, 58, 31, 15, 12,
                                17, 88, 12, 137, 4,
                                36, 10, 36, 4, 36};
    const std::vector blues = {131, 136, 22, 15, 68,
                                226, 140, 28, 213, 64,
                                106, 1, 118, 64, 217};

    std::stringstream inputFilePathStream;
    inputFilePathStream << TEST_IMAGES_INPUT_DIRPATH << "testImage.jpg";
    const std::string inputFilePath = inputFilePathStream.str();

    const auto img = imageReader->loadRGBImage(inputFilePath);

    ASSERT_NE(img, nullptr);
    EXPECT_EQ(img->getHeight(), height);
    EXPECT_EQ(img->getWidth(), width);
    ASSERT_EQ(img->getReds().size(), width * height);
    ASSERT_EQ(img->getGreens().size(), width * height);
    ASSERT_EQ(img->getBlues().size(), width * height);
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            EXPECT_EQ(img->getReds()[y * width + x], reds[y * width + x]);
            EXPECT_EQ(img->getGreens()[y * width + x], greens[y * width + x]);
            EXPECT_EQ(img->getBlues()[y * width + x], blues[y * width + x]);
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
    const std::vector<uint8_t> someReds(width * height, 0);
    const std::vector<uint8_t> someGreens(width * height, 0);
    const std::vector<uint8_t> someBlues(width * height, 0);
    const Image testImage(width, height, someReds, someGreens, someBlues);

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
    const std::vector<uint8_t> someReds(width * height, 0);
    const std::vector<uint8_t> someGreens(width * height, 0);
    const std::vector<uint8_t> someBlues(width * height, 0);
    const Image testImage(width, height, someReds, someGreens, someBlues);

    const std::string outputFilePath = "this/path/doesnt/exist/testImage.jpg";
    std::filesystem::remove_all(outputFilePath);

    EXPECT_THROW(imageReader->saveJPGImage(testImage, outputFilePath), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(outputFilePath));
}