#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <array>

#include "../src/Image.h"
#include "../src/Kernel.h"
#include "../src/ImageProcessing.h"

class ImageProcessingTest : public ::testing::Test {
protected:
    const unsigned int height = 3;
    const unsigned int width = 5;
    Pixel rgb00, rgb01, rgb02, rgb03, rgb04;
    Pixel rgb10, rgb11, rgb12, rgb13, rgb14;
    Pixel rgb20, rgb21, rgb22, rgb23, rgb24;
    Image* imageToProcess = nullptr;

    void SetUp() override {
        rgb00 = Pixel(120, 0, 130);
        rgb01 = Pixel(23, 58, 135);
        rgb02 = Pixel(44, 30, 20);
        rgb03 = Pixel(123, 15, 15);
        rgb04 = Pixel(1, 12, 68);
        const std::vector row0 = {rgb00, rgb01, rgb02, rgb03, rgb04};
        rgb10 = Pixel(1, 17, 225);
        rgb11 = Pixel(19, 89, 139);
        rgb12 = Pixel(67, 12, 29);
        rgb13 = Pixel(88, 137, 213);
        rgb14 = Pixel(81, 3, 64);
        const std::vector row1 = {rgb10, rgb11, rgb12, rgb13, rgb14};
        rgb20 = Pixel(43, 38, 106);
        rgb21 = Pixel(100, 10, 0);
        rgb22 = Pixel(215, 35, 120);
        rgb23 = Pixel(10, 4, 65);
        rgb24 = Pixel(90, 36, 217);
        const std::vector row2 = {rgb20, rgb21, rgb22, rgb23, rgb24};

        imageToProcess = new Image(width, height, std::vector{row0, row1, row2});
    }

    void TearDown() override {
        delete imageToProcess;
        imageToProcess = nullptr;
    }
};


TEST_F(ImageProcessingTest, testConvolutionWhenValuesAreInRange) {
    constexpr unsigned int order = 3;
    const std::string kernelName = "inRangeKernel";
    const Kernel kernel(kernelName, order, std::vector<float> {  0.025, 0.1, 0.025,
                                                                      0.1, 0.5, 0.1,
                                                                    0.025, 0.1, 0.025   });
    constexpr unsigned int heightConvoluted = 1;
    constexpr unsigned int widthConvoluted = 3;
    constexpr std::array<uint8_t, widthConvoluted> redsConvoluted = {39, 76, 80};
    constexpr std::array<uint8_t, widthConvoluted> greensConvoluted = {56, 37, 74};
    constexpr std::array<uint8_t, widthConvoluted> bluesConvoluted = {117, 69, 134};

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::convolution(*imageToProcess, kernel);

    EXPECT_EQ(imageProcessed->getHeight(), heightConvoluted);
    EXPECT_EQ(imageProcessed->getWidth(), widthConvoluted);
    ASSERT_EQ(imageProcessed->getData().size(), heightConvoluted);
    ASSERT_EQ(imageProcessed->getData()[0].size(), widthConvoluted);
    for (unsigned int i = 0; i < widthConvoluted; i++) {
        EXPECT_EQ(imageProcessed->getData()[0][i].getR(), redsConvoluted[i]);
        EXPECT_EQ(imageProcessed->getData()[0][i].getG(), greensConvoluted[i]);
        EXPECT_EQ(imageProcessed->getData()[0][i].getB(), bluesConvoluted[i]);
    }
}

TEST_F(ImageProcessingTest, testConvolutionWhenValuesAreNegative) {
    // image must contain at least one not-zero element for each channel
    std::vector<uint8_t> reds;
    std::vector<uint8_t> greens;
    std::vector<uint8_t> blues;
    for (auto row : imageToProcess->getData()) {
        std::transform(row.begin(), row.end(), std::back_inserter(reds), [](const Pixel& pixel){return pixel.getR();});
        std::transform(row.begin(), row.end(), std::back_inserter(greens), [](const Pixel& pixel){return pixel.getG();});
        std::transform(row.begin(), row.end(), std::back_inserter(blues), [](const Pixel& pixel){return pixel.getB();});
    }
    ASSERT_THAT(reds, testing::Contains(testing::Not(0)));
    ASSERT_THAT(greens, testing::Contains(testing::Not(0)));
    ASSERT_THAT(blues, testing::Contains(testing::Not(0)));
    
    constexpr unsigned int order = 3;
    const std::string kernelName = "negativeKernel";
    constexpr float negativeWeight = -1.0;
    const Kernel negativeKernel(kernelName, order, std::vector(order * order, negativeWeight));

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::convolution(*imageToProcess, negativeKernel);

    for (unsigned int i = 0; i < 3; i++) {
        EXPECT_EQ(imageProcessed->getData()[0][i].getR(), 0);
        ASSERT_EQ(imageProcessed->getData()[0][i].getG(), 0);
        ASSERT_EQ(imageProcessed->getData()[0][i].getB(), 0);
    }
}

TEST_F(ImageProcessingTest, testConvolutionWhenValuesAreOutOfRange) {
    // image must contain at least one not-zero element for each channel
    std::vector<uint8_t> reds;
    std::vector<uint8_t> greens;
    std::vector<uint8_t> blues;
    for (auto row : imageToProcess->getData()) {
        std::transform(row.begin(), row.end(), std::back_inserter(reds), [](const Pixel& pixel){return pixel.getR();});
        std::transform(row.begin(), row.end(), std::back_inserter(greens), [](const Pixel& pixel){return pixel.getG();});
        std::transform(row.begin(), row.end(), std::back_inserter(blues), [](const Pixel& pixel){return pixel.getB();});
    }
    ASSERT_THAT(reds, testing::Contains(testing::Not(0)));
    ASSERT_THAT(greens, testing::Contains(testing::Not(0)));
    ASSERT_THAT(blues, testing::Contains(testing::Not(0)));

    constexpr unsigned int order = 3;
    const std::string kernelName = "outOfRangeKernel";
    constexpr float outOfRangeWeight = 256.0;
    const Kernel outOfRangeKernel(kernelName, order, std::vector(order * order, outOfRangeWeight));

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::convolution(*imageToProcess, outOfRangeKernel);

    for (unsigned int i = 0; i < 3; i++) {
        EXPECT_EQ(imageProcessed->getData()[0][i].getR(), 255);
        EXPECT_EQ(imageProcessed->getData()[0][i].getG(), 255);
        EXPECT_EQ(imageProcessed->getData()[0][i].getB(), 255);
    }
}


TEST_F(ImageProcessingTest, testExtendEdgeWithPositivePadding) {
    constexpr unsigned int padding = 1;
    constexpr unsigned int heightExtended = 5;
    constexpr unsigned int widthExtended = 7;
    const std::vector row0 = {rgb00, rgb00, rgb01, rgb02, rgb03, rgb04, rgb04};
    const std::vector row1 = {rgb00, rgb00, rgb01, rgb02, rgb03, rgb04, rgb04};
    const std::vector row2 = {rgb10, rgb10, rgb11, rgb12, rgb13, rgb14, rgb14};
    const std::vector row3 = {rgb20, rgb20, rgb21, rgb22, rgb23, rgb24, rgb24};
    const std::vector row4 = {rgb20, rgb20, rgb21, rgb22, rgb23, rgb24, rgb24};
    const std::vector pixelsProcessed = {row0, row1, row2, row3, row4};

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::extendEdge(*imageToProcess, padding);

    EXPECT_EQ(imageProcessed->getHeight(), heightExtended);
    EXPECT_EQ(imageProcessed->getWidth(), widthExtended);
    ASSERT_EQ(imageProcessed->getData().size(), heightExtended);
    ASSERT_EQ(imageProcessed->getData()[0].size(), widthExtended);
    for (unsigned int j = 0; j < heightExtended; j++) {
        for (unsigned int i = 0; i < widthExtended; i++) {
            EXPECT_EQ(imageProcessed->getData()[j][i].getR(), pixelsProcessed[j][i].getR());
            EXPECT_EQ(imageProcessed->getData()[j][i].getG(), pixelsProcessed[j][i].getG());
            EXPECT_EQ(imageProcessed->getData()[j][i].getB(), pixelsProcessed[j][i].getB());
        }
    }
}

TEST_F(ImageProcessingTest, testExtendEdgeWithZeroPadding) {
    constexpr unsigned int padding = 0;
    constexpr unsigned int heightExtended = 3;
    constexpr unsigned int widthExtended = 5;
    const std::vector row0 = {rgb00, rgb01, rgb02, rgb03, rgb04};
    const std::vector row1 = {rgb10, rgb11, rgb12, rgb13, rgb14};
    const std::vector row2 = {rgb20, rgb21, rgb22, rgb23, rgb24};
    const std::vector pixelsProcessed = {row0, row1, row2};

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::extendEdge(*imageToProcess, padding);

    EXPECT_EQ(imageProcessed->getHeight(), heightExtended);
    EXPECT_EQ(imageProcessed->getWidth(), widthExtended);
    ASSERT_EQ(imageProcessed->getData().size(), heightExtended);
    ASSERT_EQ(imageProcessed->getData()[0].size(), widthExtended);
    for (unsigned int j = 0; j < heightExtended; j++) {
        for (unsigned int i = 0; i < widthExtended; i++) {
            EXPECT_EQ(imageProcessed->getData()[j][i].getR(), pixelsProcessed[j][i].getR());
            EXPECT_EQ(imageProcessed->getData()[j][i].getG(), pixelsProcessed[j][i].getG());
            EXPECT_EQ(imageProcessed->getData()[j][i].getB(), pixelsProcessed[j][i].getB());
        }
    }
    EXPECT_NE(imageToProcess, imageProcessed.get());
}