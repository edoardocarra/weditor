cmake_minimum_required (VERSION 3.9)
project (Editor)

include_directories(external)
include_directories(external/glm)
include_directories(external/glfw/include)

SET (CMAKE_CXX_STANDARD 17)
SET (CMAKE_CXX_STANDARD_REQUIRED ON)
SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET (CMAKE_C_COMPILER_WORKS 1)
SET (CMAKE_CXX_COMPILER_WORKS 1)
SET (CMAKE_BUILD_TYPE "Release")
SET (CMAKE_FIND_LIBRARY_SUFFIXES ".lib")

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Vulkan REQUIRED)
find_library(GLFW_LIBRARY NAMES glfw3dll PATHS external/glfw/lib-vc2015)

add_executable(${PROJECT_NAME} source/editor.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_include_directories(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan ${GLFW_LIBRARY})