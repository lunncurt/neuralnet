cmake_minimum_required(VERSION 3.16...4.1)
project(neuralnet LANGUAGES CXX)

set(SOURCE_FILES
    src/main.cpp
    src/classes/neural.cpp
    src/classes/layer.cpp
    src/classes/utils.cpp)

add_executable(out ${SOURCE_FILES})

target_compile_features(out PRIVATE cxx_std_20)
set_target_properties(out PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)

target_include_directories(out PRIVATE src)
target_compile_definitions(out PRIVATE
    NEURALNET_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Eigen3 REQUIRED)
if(TARGET Eigen3::Eigen)
    target_link_libraries(out PRIVATE Eigen3::Eigen)
else()
    target_include_directories(out PRIVATE ${EIGEN3_INCLUDE_DIRS})
endif()

find_package(SFML 3 QUIET COMPONENTS Graphics CONFIG)
if(SFML_FOUND)
    target_link_libraries(out PRIVATE SFML::Graphics)
else()
    find_package(SFML 2.5 REQUIRED COMPONENTS graphics window system)
    target_include_directories(out PRIVATE ${SFML_INCLUDE_DIRS})
    target_link_libraries(out PRIVATE ${SFML_LIBRARIES})
endif()
