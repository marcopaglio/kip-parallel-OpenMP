#include <iostream>
#include <sstream>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <thread>

#include "timer/HighResolutionTimer.h"
#include "image/Image.h"
#include "kernel/Kernel.h"
#include "image/reader/STBImageReader.h"
#include "processing/ImageProcessing.h"
#include "kernel/KernelFactory.h"
#include "timer/SteadyTimer.h"
#include "timer/Timer.h"

namespace KernelInfos {
    enum KernelTypes {
        box_blur,
        edge_detection
    };

    static KernelTypes selectedTypes[] = {box_blur};
    constexpr unsigned int numSelectedTypes = sizeof(selectedTypes) / sizeof(KernelTypes);

    static unsigned int selectedOrders[] = {7, 13, 19, 25};
    constexpr unsigned int numSelectedOrders = sizeof(selectedOrders) / sizeof(unsigned int);
}

int main() {
    constexpr unsigned int numReps = 3;
    constexpr unsigned int minImageQuality = 4;
    constexpr unsigned int maxImageQuality = 7;
    constexpr unsigned int numImageQuality = 3;
    const std::string cvsName = "kip_parallel_openMP.csv";
    int numThreads = 16;

    try {
        // setup num threads
#ifdef _OPENMP
        if (int max_threads = omp_get_max_threads(); numThreads > max_threads) numThreads = max_threads;
        omp_set_num_threads(numThreads);
#else
        numThreads = 1;
#endif
        std::cout << "Using " << numThreads << " thread(s)." << std::endl << std::endl;

        // setup timer
        std::unique_ptr<Timer> timer;
        if constexpr (std::chrono::high_resolution_clock::is_steady)
            timer = std::make_unique<HighResolutionTimer>();
        else
            timer = std::make_unique<SteadyTimer>();

        // setup csv
        std::ofstream csvFile(cvsName);
        csvFile << "ImageName,ImageDimension,KernelName,KernelDimension,NumReps,TotalTime_s,TimePerRep_s" << "\n";

        // setup image reader
        STBImageReader imageReader{};
        std::stringstream fullPathStream;

        for (unsigned int imageQuality = minImageQuality; imageQuality <= maxImageQuality; imageQuality++) {
            for (unsigned int imageNum = 1; imageNum <= numImageQuality; imageNum++) {
                const std::string imageName = std::to_string(imageQuality) + "K-" + std::to_string(imageNum);

                // load img
                fullPathStream << IMAGES_INPUT_DIRPATH << imageName << ".jpg";
                const auto img = imageReader.loadRGBImage(fullPathStream.str());
                std::cout << "Image " << imageName << " (" << img->getWidth() << "x" << img->getHeight() <<
                    ") loaded from: " << fullPathStream.str() << std::endl;
                fullPathStream.str(std::string());

                for (const unsigned int order : KernelInfos::selectedOrders) {
                    // enlargement
                    const auto extendedImage = ImageProcessing::extendEdge(*img, (order - 1) / 2);
                    std::cout << "Image "  << imageName << " enlarged to " <<
                        extendedImage->getWidth() << "x" << extendedImage->getHeight() << std::endl;

                    for (const auto kernelType : KernelInfos::selectedTypes) {
                        // create kernel
                        std::unique_ptr<Kernel> kernel;
                        switch (kernelType) {
                            case KernelInfos::box_blur:
                                kernel = KernelFactory::createBoxBlurKernel(order);
                                break;
                            case KernelInfos::edge_detection:
                                kernel = KernelFactory::createEdgeDetectionKernel(order);
                                break;
                            default:
                                throw std::invalid_argument("Invalid kernel type");
                        }
                        std::cout << "Kernel \"" << kernel->getName() << "\" " <<
                            kernel->getOrder() << "x" << kernel->getOrder() << " created." << std::endl;

                        // transform
                        const std::chrono::duration<double> wall_clock_time_start = timer->now();
                        std::unique_ptr<Image> outputImage;
                        for (unsigned int rep = 0; rep < numReps; rep++)
                            outputImage = ImageProcessing::convolution(*extendedImage, *kernel);
                        const std::chrono::duration<double> wall_clock_time_end = timer->now();
                        const std::chrono::duration<double> wall_clock_time_duration = wall_clock_time_end - wall_clock_time_start;
                        std::cout << "Image processed " << numReps << " times in " << wall_clock_time_duration.count() <<
                            " seconds [Wall Clock] with an average of " << wall_clock_time_duration.count() / numReps <<
                            " seconds [Wall Clock] per repetition." << std::endl;

                        // save
                        fullPathStream << IMAGES_OUTPUT_DIRPATH << imageName <<
                            "_" << kernel->getName() << kernel->getOrder() << ".jpg";
                        imageReader.saveJPGImage(*outputImage, fullPathStream.str());
                        std::cout << "Image " << outputImage->getWidth() << "x" << outputImage->getHeight() <<
                            " saved at: " << fullPathStream.str() << std::endl << std::endl;
                        fullPathStream.str(std::string());

                        // csv record
                        csvFile << imageName << ","
                                << img->getWidth() << "x" << img->getHeight() << ","
                                << kernel->getName() << ","
                                << order << ","
                                << numReps << ","
                                << wall_clock_time_duration.count() << ","
                                << wall_clock_time_duration.count() / numReps
                                << "\n";
                    }

                    // idle time
                    if (imageQuality != maxImageQuality ||
                        order != KernelInfos::selectedOrders[KernelInfos::numSelectedOrders - 1]) {
                        unsigned int idleTime = KernelInfos::numSelectedTypes * (imageQuality + order / 2) / 2;
                        std::cout << "Take a pause of " << idleTime << " seconds... ";
                        std::this_thread::sleep_for(std::chrono::seconds(idleTime));
                        std::cout << "finished!" << std::endl << std::endl;
                    }
                }
            }
        }
        csvFile.close();
        std::cout << "Data saved on " << CMAKE_BINARY_DIR << "/" << cvsName << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}