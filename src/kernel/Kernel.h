#ifndef KERNEL_H
#define KERNEL_H
#include <string>
#include <vector>


/**
 * Represents a kernel used in image processing operations, providing a name,
 * order, and associated weights.
 *
 *  This class is immutable once constructed.
 */
class Kernel {
public:
    /**
     * Constructs a Kernel object with the specified name, order, and weights.
     *
     * @param name The name of the kernel as a string.
     * @param order The order of the kernel.
     * @param weights A vector of floating-point numbers representing the weights associated with the kernel.
     */
    explicit Kernel(std::string name, unsigned int order, const std::vector<float>& weights);

    /**
     * Default destructor.
     */
    ~Kernel();

    /**
     * Retrieves the name of the kernel.
     *
     * @return A string representing the name of the kernel.
     */
    [[nodiscard]] std::string getName() const;

    /**
     * Retrieves the order of the kernel.
     *
     * @return The order of the kernel as an unsigned integer.
     */
    [[nodiscard]] unsigned int getOrder() const;

    /**
     * Retrieves the weights of the kernel.
     *
     * @return A vector of floating-point numbers representing the weights of the kernel.
     */
    [[nodiscard]] std::vector<float> getWeights() const;

private:
    /**
     * Represents the name of the kernel, i.e. a string that identifies the kernel.
     */
    std::string name;

    /**
     * Represents the order of the kernel, defining its mathematical or functional complexity.
     */
    unsigned int order;

    /**
     * Stores the weights associated with the kernel as a collection of floating-point values.
     */
    std::vector<float> weights;
};



#endif //KERNEL_H
