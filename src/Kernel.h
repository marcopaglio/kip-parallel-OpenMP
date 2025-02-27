#ifndef KERNEL_H
#define KERNEL_H
#include <string>
#include <vector>


class Kernel {
public:
    explicit Kernel(std::string name, unsigned int order, const std::vector<float>& weights);
    ~Kernel();

    [[nodiscard]] std::string getName() const;
    [[nodiscard]] unsigned int getOrder() const;
    [[nodiscard]] std::vector<float> getWeights() const;

private:
    std::string name;
    unsigned int order;
    std::vector<float> weights;
};



#endif //KERNEL_H
