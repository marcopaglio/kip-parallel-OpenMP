#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <array>

#include "image/Image.h"
#include "kernel/Kernel.h"
#include "processing/ImageProcessing.h"

class ImageProcessingTest : public ::testing::Test {
protected:
    const unsigned int height = 3;
    const unsigned int width = 5;
    std::vector<uint8_t> reds;
    std::vector<uint8_t> greens;
    std::vector<uint8_t> blues;
    Image* imageToProcess = nullptr;

    void SetUp() override {
        reds = {120, 23, 44, 123, 1,
                                1, 19, 67, 88, 81,
                                43, 100, 215, 10, 90};
        greens = {0, 58, 30, 15, 12,
                                    17, 89, 12, 137, 3,
                                    38, 10, 35, 4, 36};
        blues = {130, 135, 20, 15, 68,
                                    225, 139, 29, 213, 64,
                                    106, 0, 120, 65, 217};

        imageToProcess = new Image(width, height, reds, greens, blues);
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
    ASSERT_EQ(imageProcessed->getReds().size(), widthConvoluted * heightConvoluted);
    ASSERT_EQ(imageProcessed->getGreens().size(), widthConvoluted * heightConvoluted);
    ASSERT_EQ(imageProcessed->getBlues().size(), widthConvoluted * heightConvoluted);
    for (unsigned int i = 0; i < widthConvoluted; i++) {
        EXPECT_EQ(imageProcessed->getReds()[0 * widthConvoluted + i], redsConvoluted[i]);
        EXPECT_EQ(imageProcessed->getGreens()[0 * widthConvoluted + i], greensConvoluted[i]);
        EXPECT_EQ(imageProcessed->getBlues()[0 * widthConvoluted + i], bluesConvoluted[i]);
    }
}

TEST_F(ImageProcessingTest, testConvolutionWhenValuesAreNegative) {
    // image must contain at least one not-zero element for each channel
    ASSERT_THAT(imageToProcess->getReds(), testing::Contains(testing::Not(0)));
    ASSERT_THAT(imageToProcess->getGreens(), testing::Contains(testing::Not(0)));
    ASSERT_THAT(imageToProcess->getBlues(), testing::Contains(testing::Not(0)));
    
    constexpr unsigned int order = 3;
    const std::string kernelName = "negativeKernel";
    constexpr float negativeWeight = -1.0;
    const Kernel negativeKernel(kernelName, order, std::vector(order * order, negativeWeight));

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::convolution(*imageToProcess, negativeKernel);

    constexpr unsigned int widthConvoluted = 3;
    for (unsigned int i = 0; i < widthConvoluted; i++) {
        EXPECT_EQ(imageProcessed->getReds()[0 * widthConvoluted + i], 0);
        EXPECT_EQ(imageProcessed->getGreens()[0 * widthConvoluted + i], 0);
        EXPECT_EQ(imageProcessed->getBlues()[0 * widthConvoluted + i], 0);
    }
}

TEST_F(ImageProcessingTest, testConvolutionWhenValuesAreOutOfRange) {
    // image must contain at least one not-zero element for each channel
    ASSERT_THAT(imageToProcess->getReds(), testing::Contains(testing::Not(0)));
    ASSERT_THAT(imageToProcess->getGreens(), testing::Contains(testing::Not(0)));
    ASSERT_THAT(imageToProcess->getBlues(), testing::Contains(testing::Not(0)));

    constexpr unsigned int order = 3;
    const std::string kernelName = "outOfRangeKernel";
    constexpr float outOfRangeWeight = 256.0;
    const Kernel outOfRangeKernel(kernelName, order, std::vector(order * order, outOfRangeWeight));

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::convolution(*imageToProcess, outOfRangeKernel);

    constexpr unsigned int widthConvoluted = 3;
    for (unsigned int i = 0; i < widthConvoluted; i++) {
        EXPECT_EQ(imageProcessed->getReds()[0 * widthConvoluted + i], 255);
        EXPECT_EQ(imageProcessed->getGreens()[0 * widthConvoluted + i], 255);
        EXPECT_EQ(imageProcessed->getBlues()[0 * widthConvoluted + i], 255);
    }
}


TEST_F(ImageProcessingTest, testExtendEdgeWithPositivePadding) {
    constexpr unsigned int padding = 1;
    constexpr unsigned int heightExtended = 5;
    constexpr unsigned int widthExtended = 7;
    const std::vector redsProcessed = { 120, 120, 23, 44, 123, 1, 1,
                                        120, 120, 23, 44, 123, 1, 1,
                                        1, 1, 19, 67, 88, 81, 81,
                                        43, 43, 100, 215, 10, 90, 90,
                                        43, 43, 100, 215, 10, 90, 90    };
    const std::vector greensProcessed = {   0, 0, 58, 30, 15, 12, 12,
                                            0, 0, 58, 30, 15, 12, 12,
                                            17, 17, 89, 12, 137, 3, 3,
                                            38, 38, 10, 35, 4, 36, 36,
                                            38, 38, 10, 35, 4, 36, 36   };
    const std::vector bluesProcessed = {    130, 130, 135, 20, 15, 68, 68,
                                            130, 130, 135, 20, 15, 68, 68,
                                            225, 225, 139, 29, 213, 64, 64,
                                            106, 106, 0, 120, 65, 217, 217,
                                            106, 106, 0, 120, 65, 217, 217  };

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::extendEdge(*imageToProcess, padding);

    EXPECT_EQ(imageProcessed->getHeight(), heightExtended);
    EXPECT_EQ(imageProcessed->getWidth(), widthExtended);
    ASSERT_EQ(imageProcessed->getReds().size(), widthExtended * heightExtended);
    ASSERT_EQ(imageProcessed->getGreens().size(), widthExtended * heightExtended);
    ASSERT_EQ(imageProcessed->getBlues().size(), widthExtended * heightExtended);
    for (unsigned int j = 0; j < heightExtended; j++) {
        for (unsigned int i = 0; i < widthExtended; i++) {
            EXPECT_EQ(imageProcessed->getReds()[j * widthExtended + i], redsProcessed[j * widthExtended + i]);
            EXPECT_EQ(imageProcessed->getGreens()[j * widthExtended + i], greensProcessed[j * widthExtended + i]);
            EXPECT_EQ(imageProcessed->getBlues()[j * widthExtended + i], bluesProcessed[j * widthExtended + i]);
        }
    }
}

TEST_F(ImageProcessingTest, testExtendEdgeWithZeroPadding) {
    constexpr unsigned int padding = 0;
    constexpr unsigned int heightExtended = 3;
    constexpr unsigned int widthExtended = 5;

    const std::unique_ptr<Image> imageProcessed = ImageProcessing::extendEdge(*imageToProcess, padding);

    EXPECT_EQ(imageProcessed->getHeight(), heightExtended);
    EXPECT_EQ(imageProcessed->getWidth(), widthExtended);
    ASSERT_EQ(imageProcessed->getReds().size(), widthExtended * heightExtended);
    ASSERT_EQ(imageProcessed->getGreens().size(), widthExtended * heightExtended);
    ASSERT_EQ(imageProcessed->getBlues().size(), widthExtended * heightExtended);
    for (unsigned int j = 0; j < heightExtended; j++) {
        for (unsigned int i = 0; i < widthExtended; i++) {
            EXPECT_EQ(imageProcessed->getReds()[j * widthExtended + i], reds[j * widthExtended + i]);
            EXPECT_EQ(imageProcessed->getGreens()[j * widthExtended + i], greens[j * widthExtended + i]);
            EXPECT_EQ(imageProcessed->getBlues()[j * widthExtended + i], blues[j * widthExtended + i]);
        }
    }
    EXPECT_NE(imageToProcess, imageProcessed.get());
}