project(neuralnet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")

set(SOURCE_FILES
    src/main.cpp)

add_executable(out ${SOURCE_FILES})

find_package(SFML 2.6.1 COMPONENTS graphics system window REQUIRED)
include_directories(${SFML_INCLUDE_DIRECTORIES})
target_link_libraries(exec sfml-system sfml-graphics sfml-window)
