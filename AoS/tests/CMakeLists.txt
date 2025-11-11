include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
)
FetchContent_MakeAvailable(googletest)

set(TEST_SOURCES
        runAllTests.cpp
        PixelTest.cpp
        KernelTest.cpp
        ImageTest.cpp
        STBImageReaderTest.cpp
        ImageProcessingTest.cpp
        KernelFactoryTest.cpp
)

add_executable(kip_openMP_runTests ${TEST_SOURCES})

target_link_libraries(kip_openMP_runTests kip_openMP_lib gtest_main gmock_main)

add_compile_definitions(TEST_IMAGES_INPUT_DIRPATH="${PROJECT_SOURCE_DIR}/tests/imgs/input/")
add_compile_definitions(TEST_IMAGES_OUTPUT_DIRPATH="${PROJECT_SOURCE_DIR}/tests/imgs/output/")

add_test(NAME runAllTests COMMAND kip_openMP_runTests)