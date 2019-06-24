#ifndef EDITOR_H
#define EDITOR_H

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define WIDTH 800
#define HEIGHT 600

#include "GLM/vec4.hpp"
#include "GLM/mat4x4.hpp"
#include "GLFW/glfw3.h"
#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <cstring>

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

class Viewer {
    public:
        void run() {
            initWindow();
            initVulkan();
            mainLoop();
            cleanUp();
        }
    private:
        GLFWwindow* window;
        //The instance is the connection between your application and the Vulkan library
        VkInstance instance;
        //checks if all of the requested layers are available
        bool checkValidationLayerSupport() {
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

            std::vector<VkLayerProperties> availableLayers(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

           for (const char* layerName : validationLayers) {
               bool layerFound = false;
               for (const auto& layerProperties : availableLayers) {
                   if (strcmp(layerName, layerProperties.layerName) == 0) {
                       layerFound = true;
                       break;
                    }
                }
                if (!layerFound) {
                    return false;
                }
            }

            return true;
        }
        void createInstance() {

            if (enableValidationLayers && !checkValidationLayerSupport()) {
                throw std::runtime_error("validation layers requested, but not available!");
            }

            //information to the driver to optimize for our specific application
            VkApplicationInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            info.pApplicationName = "Instance";
            info.applicationVersion = VK_MAKE_VERSION(1,0,0);
            info.pEngineName = "No Engine";
            info.engineVersion = VK_MAKE_VERSION(1,0,0);
            info.apiVersion = VK_API_VERSION_1_0;

            //tell to the Vulkan driver which global extensions and validation 
            //layers we want to use
            
            VkInstanceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &info;

            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            } else {
                createInfo.enabledLayerCount = 0;
            }

            //Vulkan is a platform agnostic API, which means that you need an 
            //extension to interface with the window system. GLFW gives you tge require
            //extension

            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            createInfo.enabledExtensionCount = glfwExtensionCount;
            createInfo.ppEnabledExtensionNames = glfwExtensions;

            //determine the global validation layers to enable

            createInfo.enabledLayerCount = 0;

            // the handle is stored into VkInstance*
            if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
                throw std::runtime_error("failed to create vulkan instance");
            }

        }
        void initWindow() {
            glfwInit();
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
            window = glfwCreateWindow(WIDTH,HEIGHT,"Vulkan Viewer",nullptr, nullptr);
        }
        void initVulkan() {
            createInstance();
        }
        void mainLoop() {
            while(!glfwWindowShouldClose(window)) {
                glfwPollEvents();
            }
        }
        void cleanUp() {
            vkDestroyInstance(instance, nullptr);
            glfwDestroyWindow(window);
            glfwTerminate();
        }
};


#endif