#include <gtest/gtest.h>
#include "kernel/KernelFactory.h"

class BlurKernelFactoryTest : public ::testing::TestWithParam<std::pair<unsigned int, float>> {
protected:
    std::string boxBlurName = "boxBlur";
};

INSTANTIATE_TEST_SUITE_P(, BlurKernelFactoryTest,
                         testing::Values(
                             std::make_pair(3, 1./9),
                             std::make_pair(5, 1./25),
                             std::make_pair(7, 1./49)),
                        [](const testing::TestParamInfo<BlurKernelFactoryTest::ParamType>& info) {
                            std::stringstream suffixStream;
                            suffixStream << "WhenOrderIs" << info.param.first;
                            return suffixStream.str();
                        });


TEST_P(BlurKernelFactoryTest, testCreateBlurKernel) {
    const unsigned int order = GetParam().first;
    const float blurWeightValue = GetParam().second;
    const std::unique_ptr<Kernel> kernel = KernelFactory::createBoxBlurKernel(order);

    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->getName(), boxBlurName);
    EXPECT_EQ(kernel->getOrder(), order);
    ASSERT_EQ(kernel->getWeights().size(), order * order);
    for (auto weight : kernel->getWeights()) {
        EXPECT_EQ(weight, blurWeightValue);
    }
}


TEST_F(BlurKernelFactoryTest, testCreateBlurKernelWithEvenOrder) {
    constexpr unsigned int order = 4;
    EXPECT_THROW(KernelFactory::createBoxBlurKernel(order), std::invalid_argument);
}



class EdgeDetectionKernelCreatorTest :
    public ::testing::TestWithParam<std::pair<unsigned int, std::vector<float>>> {
protected:
    std::string edgeDetectionName = "edgeDetection";
};

INSTANTIATE_TEST_SUITE_P(, EdgeDetectionKernelCreatorTest,
                         testing::Values(
                         std::make_pair(3, std::vector<float>{ -1, -1, -1,
                                                                    -1, 8, -1,
                                                                   -1, -1, -1   }),
                         std::make_pair(5, std::vector<float>{-1, -1, -1, -1, -1,
                                                                  -1, -2, -2, -2, -1,
                                                                  -1, -2, 32, -2, -1,
                                                                  -1, -2, -2, -2, -1,
                                                                  -1, -1, -1, -1, -1  }),
                         std::make_pair(7, std::vector<float>{   -1, -1, -1, -1, -1, -1, -1,
                                                                     -1, -2, -2, -2, -2, -2, -1,
                                                                     -1, -2, -4, -4, -4, -2, -1,
                                                                     -1, -2, -4, 88, -4, -2, -1,
                                                                     -1, -2, -4, -4, -4, -2, -1,
                                                                     -1, -2, -2, -2, -2, -2, -1,
                                                                     -1, -1, -1, -1, -1, -1, -1  }),
                         std::make_pair(9, std::vector<float>{  -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                                -1, -2, -2, -2, -2, -2, -2, -2, -1,
                                                                -1, -2, -4, -4, -4, -4, -4, -2, -1,
                                                                -1, -2, -4, -8, -8, -8, -4, -2, -1,
                                                                -1, -2, -4, -8, 208, -8, -4, -2, -1,
                                                                -1, -2, -4, -8, -8, -8, -4, -2, -1,
                                                                -1, -2, -4, -4, -4, -4, -4, -2, -1,
                                                                -1, -2, -2, -2, -2, -2, -2, -2, -1,
                                                                -1, -1, -1, -1, -1, -1, -1, -1, -1,  })),
                        [](const testing::TestParamInfo<EdgeDetectionKernelCreatorTest::ParamType>& info) {
                            std::stringstream suffixStream;
                            suffixStream << "WhenOrderIs" << info.param.first;
                            return suffixStream.str();
                        });


TEST_P(EdgeDetectionKernelCreatorTest, testCreateEdgeDetectionKernel) {
    const unsigned int order = GetParam().first;
    const std::vector<float> weights = GetParam().second;

    const std::unique_ptr<Kernel> kernel = KernelFactory::createEdgeDetectionKernel(order);

    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->getName(), edgeDetectionName);
    EXPECT_EQ(kernel->getOrder(), order);
    ASSERT_EQ(kernel->getWeights().size(), order * order);
    for (unsigned int i = 0; i < order * order; i++) {
        EXPECT_EQ(kernel->getWeights()[i], weights[i]);
    }
}


TEST_F(EdgeDetectionKernelCreatorTest, testCreateEdgeDetectionKernelWithEvenOrder) {
    constexpr unsigned int order = 4;
    EXPECT_THROW(KernelFactory::createEdgeDetectionKernel(order), std::invalid_argument);
}