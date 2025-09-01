#include <array>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>

#include "timer/HighResolutionTimer.h"
#include "image/Image.h"
#include "kernel/Kernel.h"
#include "image/reader/STBImageReader.h"
#include "processing/ImageProcessing.h"
#include "kernel/KernelFactory.h"
#include "timer/SteadyTimer.h"
#include "timer/Timer.h"
#include "workload/Workload.h"

#ifdef _OPENMP
#include <omp.h>
#endif


int main() {
#ifdef _OPENMP
    const int maxNumThreads = omp_get_max_threads();
#else
    const int maxNumThreads = 1;
#endif
    constexpr unsigned int imageQuality = 4; // 4, 5, 6, 7
    constexpr unsigned int numImageQuality = 3;
    constexpr unsigned int order = 7; // 7, 13, 19, 25
    constexpr unsigned int numReps = 3;
    const std::string cvsName = "kip_openMP_weakScaling.csv";

    try {
        // setup timer
        std::unique_ptr<Timer> timer;
        if constexpr (std::chrono::high_resolution_clock::is_steady)
            timer = std::make_unique<HighResolutionTimer>();
        else
            timer = std::make_unique<SteadyTimer>();

        // setup csv
        std::ofstream csvFile(cvsName);
        csvFile << "ImageName,ImageDimension,KernelName,KernelDimension,TimePerRep_s,"
                   "NumThreads,UnitOfWork,WeakEfficiency,ScaledSpeedUp,Throughput_Mpix_s" << "\n";

        // setup image reader
        STBImageReader imageReader{};
        std::stringstream fullPathStream;

        std::array<double, numImageQuality> sequentialTimes = {};
        std::array<std::string, numImageQuality> basicWorkload = {};
        for (int numThreads = 1; numThreads <= maxNumThreads; numThreads<<=1) {
#ifdef _OPENMP
            omp_set_num_threads(numThreads);
#endif
            std::cerr << "Using: " << numThreads << " threads" << std::endl;

            for (unsigned int imageNum = 1; imageNum <= numImageQuality; imageNum++) {
                const std::string imageName = std::to_string(imageQuality) + "K-" + std::to_string(imageNum);

                // load img
                fullPathStream << PROJECT_SOURCE_DIR << "/imgs/input/" << imageName << ".jpg";
                const auto img = workload::loadExpandedRGBImage(
                    fullPathStream.str(), numThreads);
                std::cout << "Image " << imageName << " (" << img->getWidth() << "x" << img->getHeight() <<
                    ") loaded from: " << fullPathStream.str() << std::endl;
                fullPathStream.str(std::string());

                // enlargement
                const auto extendedImage = ImageProcessing::extendEdge(*img, (order - 1) / 2);
                std::cout << "Image "  << imageName << " enlarged to " <<
                    extendedImage->getWidth() << "x" << extendedImage->getHeight() << std::endl;

                // create kernel
                const std::unique_ptr<Kernel> kernel = KernelFactory::createBoxBlurKernel(order);
                std::cout << "Kernel \"" << kernel->getName() << "\" " << kernel->getOrder() << "x" << kernel->getOrder() <<
                    " created." << std::endl;

                // transform
                const std::chrono::duration<double> wall_clock_time_start = timer->now();
                for (unsigned int rep = 0; rep < numReps; rep++)
                    ImageProcessing::convolution(*extendedImage, *kernel);
                const std::chrono::duration<double> wall_clock_time_end = timer->now();
                const std::chrono::duration<double> wall_clock_time_duration = wall_clock_time_end - wall_clock_time_start;
                const auto timePerRep = wall_clock_time_duration.count() / numReps;
                std::cout << "Image processed " << numReps << " times in " << wall_clock_time_duration.count() << " seconds [Wall Clock]" <<
                    " with an average of " << timePerRep << " seconds [Wall Clock] per repetition." << std::endl;

                if (numThreads == 1) {
                    sequentialTimes[imageNum - 1] = timePerRep;
                    basicWorkload[imageNum - 1] = std::to_string(img->getWidth()) + "x" + std::to_string(img->getHeight());
                }
                const double weakEfficiency = sequentialTimes[imageNum - 1] / timePerRep;

                // csv record
                csvFile << imageName << ","
                        << img->getWidth() << "x" << img->getHeight() << ","
                        << kernel->getName() << ","
                        << order << ","
                        << timePerRep << ","
                        << numThreads << ","
                        << basicWorkload[imageNum - 1] << ","
                        << weakEfficiency << ","
                        << numThreads * weakEfficiency << ","
                        << img->getWidth() * img->getHeight() * 1e-6 / timePerRep
                        << "\n";
            }
        }
        csvFile.close();
        std::cout << "Data saved at " << CMAKE_BINARY_DIR << "/" << cvsName << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
