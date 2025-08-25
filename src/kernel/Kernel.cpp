#include <iostream>
#include "Kernel.h"

Kernel::Kernel(std::string name, const unsigned int order, const std::vector<float>& weights):
    name(std::move(name)), order(order), weights(weights) {}

Kernel::~Kernel() = default;

std::string Kernel::getName() const {
    return name;
}

unsigned int Kernel::getOrder() const {
    return order;
}

std::vector<float> Kernel::getWeights() const {
    return weights;
}