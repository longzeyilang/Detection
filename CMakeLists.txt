cmake_minimum_required(VERSION 3.5.1)
project(golfDetection)
SET(CMAKE_BUILD_TYPE "Release")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
message(${PROJECT_SOURCE_DIR}/../../aarch64-linux-gds)

option(USE_NEON "use neon to accumate in arm" ON)  #ON OFF
option(USE_MNN "if true use MNN framework else use ncnn framework" OFF)   #ON OFF

if(USE_MNN)
   add_definitions(-DUSE_MNN)
endif()

if (USE_NEON)
        add_definitions(-DUSE_NEON)
        set(CMAKE_SYSTEM_NAME Linux)
        set(CMAKE_SYSTEM_PROCESSOR arm)
        set(tools ${PROJECT_SOURCE_DIR}/../../aarch64-linux-gds)
        set(CMAKE_C_COMPILER ${tools}/bin/aarch64-linux-gnu-gcc)
        set(CMAKE_CXX_COMPILER ${tools}/bin/aarch64-linux-gnu-g++)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
        # find opencv、MNN、yaml-cpp
        # 以cmakelist.txt文件所在路径为当前路径
        include_directories("../../third_part/")
else()
        SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
        # find opencv、MNN、yaml-cpp
        include_directories("/gzy_mnt/third_party_x86/include")
endif(USE_NEON)

#include self header
include_directories("./include/")
file(GLOB SOURCE_FILES ./src/*.cpp)

set(LIBRARY_OUTPUT_PATH ../../../lib/)
add_library(golfDetection SHARED ${SOURCE_FILES})

if (USE_NEON)
        target_link_libraries (golfDetection   
        pthread
        ${PROJECT_SOURCE_DIR}/../../lib/libopencv_world.so
        ${PROJECT_SOURCE_DIR}/../../lib/libMNN.so
        ${PROJECT_SOURCE_DIR}/../../lib/libMNN_Express.so
        ${PROJECT_SOURCE_DIR}/../../lib/libyaml-cpp.so
        ${PROJECT_SOURCE_DIR}/../../lib/libncnn.so
        )
else()
        target_link_libraries (golfDetection   
        pthread
        /gzy_mnt/third_party_x86/lib/libMNN.so
        /gzy_mnt/third_party_x86/lib/libopencv_core.so
        /gzy_mnt/third_party_x86/lib/libopencv_imgcodecs.so
        /gzy_mnt/third_party_x86/lib/libopencv_highgui.so
        /gzy_mnt/third_party_x86/lib/libopencv_imgproc.so
        /gzy_mnt/third_party_x86/lib/libyaml-cpp.so
        )
endif(USE_NEON)
