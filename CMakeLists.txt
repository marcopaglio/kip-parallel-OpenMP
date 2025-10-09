cmake_minimum_required(VERSION 3.30)
project(kip_parallel_openMP)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" AND USE_ASAN)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    add_compile_options(/fsanitize=address /Zi)
endif()

# Base path to VTune SDK (env var o -D in cmake)
if (NOT ITT_BASE_PATH)
    if(DEFINED ENV{ITT_BASE_PATH})
        set(ITT_BASE_PATH $ENV{ITT_BASE_PATH})
    else()
        message(FATAL_ERROR "Path to VTune SDK is not specified:\nDefine ITT_BASE_PATH as environment variable or pass it to cmake via -D option")
    endif ()
endif()
set(ITT_INCLUDE_DIR "${ITT_BASE_PATH}/sdk/include")
set(ITT_LIBRARY_DIR "${ITT_BASE_PATH}/sdk/lib64")

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

add_library(kip_openMP_lib
        src/image/Pixel.cpp
        src/image/Pixel.h
        src/image/Image.cpp
        src/image/Image.h
        src/kernel/Kernel.cpp
        src/kernel/Kernel.h
        src/image/reader/ImageReader.cpp
        src/image/reader/ImageReader.h
        src/image/reader/STBImageReader.cpp
        src/image/reader/STBImageReader.h
        src/processing/ImageProcessing.cpp
        src/processing/ImageProcessing.h
        src/kernel/KernelFactory.cpp
        src/kernel/KernelFactory.h
)

add_library(kip_openMP_timer
        src/expt/timer/Timer.cpp
        src/expt/timer/Timer.h
        src/expt/timer/HighResolutionTimer.cpp
        src/expt/timer/HighResolutionTimer.h
        src/expt/timer/SteadyTimer.cpp
        src/expt/timer/SteadyTimer.h
)

add_executable(kip_openMP_main src/expt/main.cpp)
target_link_libraries(kip_openMP_main kip_openMP_lib kip_openMP_timer)

add_executable(kip_openMP_profile src/expt/profile.cpp)
target_include_directories(kip_openMP_profile PRIVATE ${ITT_INCLUDE_DIR})
target_link_directories(kip_openMP_profile PRIVATE ${ITT_LIBRARY_DIR})
target_link_libraries(kip_openMP_profile kip_openMP_lib libittnotify)

add_executable(kip_openMP_strong_scaling src/expt/strong_scaling.cpp)
target_link_libraries(kip_openMP_strong_scaling kip_openMP_lib kip_openMP_timer)

add_executable(kip_openMP_weak_scaling
        src/expt/weak_scaling.cpp
        src/expt/workload/Workload.cpp
        src/expt/workload/Workload.h
)
target_link_libraries(kip_openMP_weak_scaling kip_openMP_lib kip_openMP_timer)

find_package(OpenMP)
if(OPENMP_CXX_FOUND)
    message(STATUS "OpenMP for C++ found with CXX_VERSION=${OpenMP_CXX_VERSION}")
    target_link_libraries(kip_openMP_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

add_compile_definitions(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_compile_definitions(CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}")

enable_testing()

add_subdirectory(tests)
