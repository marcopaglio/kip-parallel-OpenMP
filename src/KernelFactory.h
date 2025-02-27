#ifndef KERNELFACTORY_H
#define KERNELFACTORY_H
#include <memory>

#include "Kernel.h"


class KernelFactory {
public:
    KernelFactory() = delete;
    ~KernelFactory() = delete;

    static std::unique_ptr<Kernel> createBlurKernel(unsigned int order);
    static std::unique_ptr<Kernel> createEdgeDetectionKernel(unsigned int order);
};



#endif //KERNELFACTORY_H
