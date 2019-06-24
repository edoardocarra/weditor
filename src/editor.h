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
        VkDebugUtilsMessengerEXT debugMessenger; //tell Vulkan about the callback function

        //To relay the debug messages back to our program we need to setup a 
        //debug messenger with a callback

        // debug callback function.  The VKAPI_ATTR and VKAPI_CALL ensure that 
        // the function has the right signature for Vulkan to call it.
        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData) {

            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
        }
        // function that will return the required list of extensions based 
        //on whether validation layers are enabled or not
        std::vector<const char*> getRequiredExtensions() {
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

            std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

            if (enableValidationLayers) {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }

            return extensions;
        }

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
                auto extensions = getRequiredExtensions();
                createInfo.enabledLayerCount = static_cast<uint32_t>(extensions.size());
                createInfo.ppEnabledLayerNames = extensions.data();
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
        VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
            auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
            if (func != nullptr) {
                return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
            } else {
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
        }

        void setupDebugMessenger() {
            if (!enableValidationLayers) return;
            //We'll need to fill in a structure with details about the messenger and its callback
            VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            createInfo.pfnUserCallback = debugCallback;
            createInfo.pUserData = nullptr; // Optional

            // create the VkDebugUtilsMessengerEXT
            if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
                throw std::runtime_error("failed to set up debug messenger!");
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
            setupDebugMessenger();
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