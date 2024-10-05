cmake_minimum_required(VERSION 3.4)
project(neuralnet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

set(SOURCE_FILES
    src/main.cpp
    src/classes/neural.cpp
    src/classes/layer.cpp
    src/classes/utils.cpp)

add_executable(out ${SOURCE_FILES})

find_package(SFML 2.6.1 COMPONENTS graphics system window REQUIRED)
include_directories(${SFML_INCLUDE_DIRECTORIES})

find_package(Eigen3 REQUIRED)
target_include_directories(out PUBLIC ${EIGEN3_INCLUDE_DIRS})

target_link_libraries(out sfml-graphics)
