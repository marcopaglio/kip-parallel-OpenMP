#ifndef WORKLOAD_H
#define WORKLOAD_H
#include <filesystem>

#include "image/Image.h"


namespace workload {
    /**
     * Do the same thing of loadRGBImage method in STBImageReader class,
     * but repeat numLoads times the loading and create a bigger RGB image.
     */
    std::unique_ptr<Image> loadExpandedRGBImage(const std::filesystem::path& filePath, unsigned int numLoads);

};



#endif //WORKLOAD_H
