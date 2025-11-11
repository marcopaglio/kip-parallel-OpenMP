#include <stdexcept>
#include "KernelFactory.h"

void checkOrderValidity(const unsigned int order) {
    if (order % 2 == 0)
        throw std::invalid_argument("Kernel order must be odd.");
}

std::unique_ptr<Kernel> createKernel(const std::string& name, unsigned int order, const std::vector<float> &weights) {
    return std::make_unique<Kernel>(name, order, weights);
}

std::unique_ptr<Kernel> KernelFactory::createBoxBlurKernel(const unsigned int order) {
    checkOrderValidity(order);

    const unsigned int num_of_values = order * order;
    const float mean =  1 / static_cast<float>(num_of_values);

    return createKernel("boxBlur", order, std::vector(num_of_values, mean));
}

std::unique_ptr<Kernel> KernelFactory::createEdgeDetectionKernel(const unsigned int order) {
    checkOrderValidity(order);

    std::vector<float> weights(order * order, -1);
    unsigned int coreWeight = order * order - 1;

    const unsigned int corePoint = order / 2;
    for (unsigned int k = 1; k < corePoint; k++) {
        const unsigned int difference = 1 << k - 1;
        for (unsigned int j = k; j < order - k; j++) {
            for (unsigned int i = k; i < order - k; i++) {
                weights[j * order + i] -= static_cast<float>(difference);
            }
        }
        coreWeight += difference * ((order - 2*k)*(order - 2*k) - 1);
    }
    weights[corePoint * order + corePoint] = static_cast<float>(coreWeight);

    return createKernel("edgeDetection", order, weights);
}