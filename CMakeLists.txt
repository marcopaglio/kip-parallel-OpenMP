cmake_minimum_required(VERSION 3.30)
project(kip_parallel_openMP)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(kip_openMP_lib
        src/Pixel.cpp
        src/Pixel.h
        src/Image.cpp
        src/Image.h
        src/Kernel.cpp
        src/Kernel.h
        src/ImageReader.cpp
        src/ImageReader.h
        src/STBImageReader.cpp
        src/STBImageReader.h
        src/ImageProcessing.cpp
        src/ImageProcessing.h
        src/KernelFactory.cpp
        src/KernelFactory.h
        src/Timer.cpp
        src/Timer.h
        src/HighResolutionTimer.cpp
        src/HighResolutionTimer.h
        src/SteadyTimer.cpp
        src/SteadyTimer.h
)

add_executable(kip_openMP_main src/main.cpp)

target_link_libraries(kip_openMP_main kip_openMP_lib)

add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

enable_testing()

add_subdirectory(tests)
