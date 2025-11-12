#ifndef KERNELFACTORY_H
#define KERNELFACTORY_H
#include <memory>

#include "Kernel.h"


/**
 * Class responsible for creating specific kernel instances.
 *
 * This is a purely static utility class, and it cannot be instantiated.
 */
class KernelFactory {
public:
    /**
     * Deleted default constructor to prevent instantiation of KernelFactory objects.
     */
    KernelFactory() = delete;

    /**
     * This destructor is explicitly deleted to ensure that KernelFactory remains a purely static utility class
     * and is not instantiated or managed in any way.
     */
    ~KernelFactory() = delete;

    /**
     * Creates a box blur kernel for image processing.
     *
     * @param order The size of the square kernel. It must be a positive odd integer.
     * @return A unique pointer to a Kernel object configured for blurring.
     * @throws std::invalid_argument if the provided order is even.
     */
    static std::unique_ptr<Kernel> createBoxBlurKernel(unsigned int order);


    /**
     * Creates an edge detection kernel for image processing.
     *
     * @param order The size of the square kernel. It must be a positive odd integer.
     * @return A unique pointer to a Kernel object configured for edge detection.
     * @throws std::invalid_argument if the provided order is even.
     */
    static std::unique_ptr<Kernel> createEdgeDetectionKernel(unsigned int order);
};



#endif //KERNELFACTORY_H
