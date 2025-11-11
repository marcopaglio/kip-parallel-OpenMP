#include <gtest/gtest.h>
#include "kernel/Kernel.h"


TEST(KernelTest, testConstructor) {
    const std::string name = "kernelTestConstructor";
    constexpr unsigned int order = 3;
    constexpr float weightValue = 0.5;

    const Kernel kernel(name, order, std::vector (order * order, weightValue));

    EXPECT_EQ(kernel.getName(), name);
    EXPECT_EQ(kernel.getOrder(), order);
    EXPECT_EQ(kernel.getWeights().size(), order * order);
    for (auto weight : kernel.getWeights())
        EXPECT_EQ(weight, weightValue);
}