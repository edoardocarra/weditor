#ifndef EDITOR_H
#define EDITOR_H

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS

#define WIDTH 800
#define HEIGHT 600

#include "GLFW/glfw3.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <chrono>

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    
    //A vertex binding describes at which rate to load data from memory throughout the vertices
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    // An attribute description struct describes how to extract a vertex attribute 
    //from a chunk of vertex data originating from a binding description
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
        //position attribute
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        //color attribute
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }

};

//descriptor
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

/* The stages that the current frame has already progressed through are idle 
and could already be used for a next frame*/
const int MAX_FRAMES_IN_FLIGHT = 2;

//validation layers are optional components that hook into Vulkan function calls to apply additional operations.
// Once defined , they currently have no way to relay the debug messages back to our program.
//To receive those messages we have to set up a debug messenger with a callback,
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

/* Vulkan does not have the concept of a "default framebuffer", hence it requires an infrastructure 
that will own the buffers we will render to before we visualize them on the screen. 
This infrastructure is known as the swap chain and must be created explicitly in Vulkan. 
The swap chain is essentially a queue of images that are waiting to be presented to the screen. 

image representation is tied to the windows system, so it is not part of the Vulkan Core.
We need to enable enable the VK_KHR_swapchain device extension after querying for its support.
*/
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    // we need to extend isDeviceSuitable to ensure that a device 
    // can present images to the surface we created
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

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
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        //After selecting a physical device to use we 
        //need to set up a logical device to interface with it.
        VkDevice device;
        //queue with graphics capabilities
        VkQueue graphicsQueue;
        //create the presentation queue 
        VkQueue presentQueue;
        //Vulkan is a platform agnostic API, it can not interface directly 
        //with the window system on its own.To establish the connection between Vulkan 
        //and the window system to present results to the screen, 
        //we need to use the WSI (Window System Integration) extensions.
        VkSurfaceKHR surface;
        VkSwapchainKHR swapChain;
        // retrieving the handles of the VkImages in the swap chain
        std::vector<VkImage> swapChainImages;
        VkFormat swapChainImageFormat;
        VkExtent2D swapChainExtent;
        // To use any VkImage, including those in the swap chain, in the render pipeline we have to create 
        // a VkImageView object. An image view is quite literally a view into an image.
        // It describes how to access the image and which part of the image to access
        std::vector<VkImageView> swapChainImageViews;

        //All of the descriptor bindings are combined into this object
        VkDescriptorSetLayout descriptorSetLayout;
        /* if we want to change the behaviour of the shader at drawing time without recreating it, we can user 
        uniform values in the shader. Those uniforms are used to pass the transformation matrix
        in the vertex shader, or to create texture samples in the fragment shader. 
        These uniforms needs to be specified during the pipeline creation by creating a pipeline layout struct */
        VkPipelineLayout pipelineLayout;
        VkRenderPass renderPass;

        VkPipeline graphicsPipeline;

        /* 
        - The attachments specified during render pass creation are bound by wrapping them into a VkFramebuffer object
        - A framebuffer object references all of the VkImageView objects that represent the attachments
        - the image that we have to use for the attachment depends on which image the swap chain returns 
        when we retrieve one for presentation
        - we have to create a framebuffer for all of the images in the swap chain and use the one that 
        corresponds to the retrieved image at drawing time.
        */
        std::vector<VkFramebuffer> swapChainFramebuffers;

        //we have to record all of the operations you want to perform in command buffer objects
        VkCommandPool commandPool;
/*      Because one of the drawing commands involves binding the right VkFramebuffer, 
        we'll actually have to record a command buffer for every image in the swap chain once again.  */
        std::vector<VkCommandBuffer> commandBuffers;

        //Each frame should have its own set of semaphores
        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        //To use the right pair of semaphores every time, we need to keep track of the current frame
        size_t currentFrame = 0;

/*       To perform CPU-GPU synchronization, we build a fence for each frame
        Fences are mainly designed to synchronize your application itself with rendering operation,
        whereas semaphores are used to synchronize operations within or across command queues.  */
        std::vector<VkFence> inFlightFences;
        //member variable that flags that a resize has happened
        bool framebufferResized = false;
        
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;

        //buffer that contains the UBO data for the shader

        /* 
        copy new data to the uniform buffer every frame, so it doesn't really make 
        any sense to have a staging buffer. It would just add extra overhead in this 
        case and likely degrade performance instead of improving it.
         */
        std::vector<VkBuffer> uniformBuffers;
        std::vector<VkDeviceMemory> uniformBuffersMemory;

        VkDescriptorPool descriptorPool;
        //hold the descriptor set handles
        std::vector<VkDescriptorSet> descriptorSets;

        void initWindow() {
            glfwInit();
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            window = glfwCreateWindow(WIDTH,HEIGHT,"Vulkan",nullptr, nullptr);
            //using this glfw function we set inside THIS an arbitray pointer to a GLFWwindow
            //so inside the framebufferResizeCallback i can reference to GLFWwindow*
            glfwSetWindowUserPointer(window, this);
            glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        }
        void initVulkan() {
            createInstance();
            setupDebugMessenger();
            // The window surface needs to be created right after the instance creation, 
            // because it can actually influence the physical device selection. 
            createSurface(); 
            pickPhysicalDevice();
            createLogicalDevice();
            //must be created after logical device creation
            createSwapChain();
            createImageViews();
            createRenderPass();
            //We need to provide details about every descriptor binding used in the shaders for pipeline creation
            createDescriptorSetLayout();
            createGraphicsPipeline();
            createFramebuffers();
            createCommandPool();
            // buffers do not automatically allocate memory for themselves. We must do that by our own
            createVertexBuffer();
            createIndexBuffer();
            createUniformBuffers();
            createDescriptorPool();
            createDescriptorSets();
            createCommandBuffers();
            createSyncObjects();
        }
        void mainLoop() {
            while(!glfwWindowShouldClose(window)) {
                glfwPollEvents();
                drawFrame();
            }
            //wait for the logical device to finish operations before exiting and destroy the window
            vkDeviceWaitIdle(device);
        }
        void cleanUp() {
            cleanupSwapChain();

            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            vkDestroyBuffer(device, indexBuffer, nullptr);
            vkFreeMemory(device, indexBufferMemory, nullptr);

            vkDestroyBuffer(device, vertexBuffer, nullptr);
            vkFreeMemory(device, vertexBufferMemory, nullptr);
            
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
                vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
                vkDestroyFence(device, inFlightFences[i], nullptr);
            }

            vkDestroyCommandPool(device, commandPool, nullptr);

            vkDestroyDevice(device, nullptr);

            if (enableValidationLayers) {
                DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
            }

            vkDestroySurfaceKHR(instance, surface, nullptr);
            vkDestroyInstance(instance, nullptr);

            glfwDestroyWindow(window);

            glfwTerminate();
        }

        //updates the uniform buffer with a new transformation every frame
        void createUniformBuffers() {
            VkDeviceSize bufferSize = sizeof(UniformBufferObject);

            uniformBuffers.resize(swapChainImages.size());
            uniformBuffersMemory.resize(swapChainImages.size());

            for (size_t i = 0; i < swapChainImages.size(); i++) {
                createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            }
        }

        //Descriptor sets can't be created directly, they must be allocated from a pool like command buffers
        void createDescriptorPool() {
            //describe which descriptor types our descriptor sets are going to contain and how many of them
            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSize.descriptorCount = static_cast<uint32_t>(swapChainImages.size());

            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            //specify the maximum number of descriptor sets that may be allocated
            poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());;

            if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
                throw std::runtime_error("failed to create descriptor pool!");
            }
        }

        //We need to provide details about every descriptor binding used in the shaders for pipeline creation, as we did for 
        //location index
        void createDescriptorSetLayout() {
            VkDescriptorSetLayoutBinding uboLayoutBinding = {};
            //binding used in the shader and the type of descriptor, which is a uniform buffer object
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            uboLayoutBinding.descriptorCount = 1;
            //in which shader stages the descriptor is going to be referenced
            uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            //image sampling related descriptors
            uboLayoutBinding.pImmutableSamplers = nullptr;

            //create the VkDescriptorSetLayout object for descriptor bindings
            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 1;
            layoutInfo.pBindings = &uboLayoutBinding;

            if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
                throw std::runtime_error("failed to create descriptor set layout!");
            }


        }

        void createIndexBuffer() {
            VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, indices.data(), (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);

            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

            copyBuffer(stagingBuffer, indexBuffer, bufferSize);

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }

        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
            VkBufferCreateInfo bufferInfo = {};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = size;
            // for which purposes the data in the buffer is going to be used
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to create buffer!");
            }
            //assigning memory to the buffer
            VkMemoryRequirements memRequirements;
            //query for the buffer memory requirements
            vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

            VkMemoryAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

            if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate buffer memory!");
            }

            //associate the memory with the buffer 
            /* 
            Since this memory is allocated specifically for this the vertex buffer, the offset is simply 0. 
            If the offset is non-zero, then it is required to be divisible by memRequirements.alignment.
            */
            vkBindBufferMemory(device, buffer, bufferMemory, 0);
        }

        /* The most optimal memory has the VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT flag and 
        is usually not accessible by the CPU on dedicated graphics cards 
        
        we're going to create two vertex buffers: 
        - One staging buffer in CPU accessible memory to upload the data from the vertex array to.
        - The final vertex buffer in device local memory.
        */

        void createVertexBuffer() {
            VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

            //stagingBuffer with stagingBufferMemory for mapping and copying the vertex data.
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            /* 
            The driver may not immediately copy the data into the buffer memory, 
            for example because of caching. It is also possible that writes to the 
            buffer are not visible in the mapped memory yet. To resolve that,
            use a memory heap that is host coherent, indicated with VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            */
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            //We're using a new stagingBuffer with stagingBufferMemory for mapping and copying the vertex data
            void* data;
            //access a region of the specified memory resource defined by an offset and size
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            //memcpy the vertex data to the mapped memory and unmap it again using vkUnmapMemory
            memcpy(data, vertices.data(), (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);
                    
            /* 
            we can copy data from the stagingBuffer to the vertexBuffer. 
            We have to indicate that we intend to do that by specifying the transfer source flag 
            for the stagingBuffer and the transfer destination flag for the vertexBuffer, 
            along with the vertex buffer usage flag. */
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
            copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);

        }

        void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
            //Memory transfer operations are executed using command buffers,
            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = commandPool;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer;
            vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
            //Start recording the command buffer:
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);

            VkBufferCopy copyRegion = {};
            copyRegion.size = size;
            vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

            vkEndCommandBuffer(commandBuffer);

            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            /* Unlike the draw commands, there are no events we need to wait on this time. 
            We just want to execute the transfer on the buffers immediately 
            Wait for the transfer queue to become idle with vkQueueWaitIdle.
            */
            vkQueueWaitIdle(graphicsQueue);

            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        }

        /*
        Graphics cards can offer different types of memory to allocate from.
        We need to combine the requirements of the buffer and our own application 
        requirements to find the right type of memory to use*/
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
            VkPhysicalDeviceMemoryProperties memProperties;
            //query info about the available types of memory
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
            /* 
            Memory heaps are distinct memory resources like dedicated VRAM and swap space 
            in RAM for when VRAM runs out. The different types of memory exist within these heaps
            */
            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
                /*
                we can find the index of a suitable memory type by simply iterating over 
                them and checking if the corresponding bit is set to 
                
                We also need to be able to write our vertex data to that memory. VkMemoryType structures in 
                memProperties specify heap and properties for each type of memory. One of the property is if 
                the memory can be mapped, so we can write to it from the cpu. Since we need to write 
                vertex buffer in this memory, we need to check for that property too.
                */
                if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                        return i;
                }
            }
            throw std::runtime_error("failed to find suitable memory type!");

        }

/*      Detect resizes with the GLFW framework, creating a callback
        We create static function as a callback is because GLFW does not know how to properly call 
        a member function with the right this pointer to our Viewer instance. */
        static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
            auto app = reinterpret_cast<Viewer*>(glfwGetWindowUserPointer(window));
            app->framebufferResized = true;
        }

        void createSyncObjects() {
            imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

            VkSemaphoreCreateInfo semaphoreInfo = {};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

/*          By default, fences are created in the unsignaled state.
            we can change the fence creation to initialize it in the signaled state as 
            if we had rendered an initial frame that finished */
            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;


            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                    vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                    vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {

                    throw std::runtime_error("failed to create synchronization objects for a frame!");
                }
            }
        }

        /*
        - Acquire an image from the swap chain
        - Execute the command buffer with that image as attachment in the framebuffer
        - Return the image to the swap chain for presentation

        Each of these events is set in motion using a single function call, but they are executed asynchronously
        We want to synchronize the queue operations of draw commands and presentation
        */
        void drawFrame() {
            //takes an array of fences and waits for either any or all of them to be signaled before returning
            vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
            vkResetFences(device, 1, &inFlightFences[currentFrame]);

            // acquire an image from the swap chain
            uint32_t imageIndex;
            /* 
            vkAcquireNextImageKHR return two kind of information which can tell us if 
            we need to restore the swap chain:
            - VK_ERROR_OUT_OF_DATE_KHR The swap chain has become incompatible with the surface and can no longer be used for rendering (when you resize window)
            - VK_SUBOPTIMAL_KHR the surface properties are no longer matched exactly. 
            */
            VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapChain();
                return;
            } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
                throw std::runtime_error("failed to acquire swap chain image!");
            }

            updateUniformBuffer(imageIndex);

            //Queue submission and synchronization 
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            //We want to wait with writing colors to the image until it's available,
            VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
            VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;

            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

            VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            // if we abort drawing in the PREVIOUS cycle, 
            // at this point then the fence will never have been submitted with vkQueueSubmit, 
            // being in THIS cycle in a possible unexpected state. So we reset fences here.
            vkResetFences(device, 1, &inFlightFences[currentFrame]);

            //We can now submit the command buffer to the graphics queue
            if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit draw command buffer!");
            }

/*          The last step of drawing a frame is submitting the result back to the swap chain to 
            have it eventually show up on the screen */
            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            //which semaphores to wait on before presentation can happen
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;
            //the swap chains to present images to and the index of the image for each swap chain
            VkSwapchainKHR swapChains[] = {swapChain};
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapChains;
            presentInfo.pImageIndices = &imageIndex;
/*          Results. It allows you to specify an array of VkResult values to check for every 
            individual swap chain if presentation was successful. It's not necessary if you're only 
            using a single swap chain, because you can simply use the return value of the present function */
            presentInfo.pResults = nullptr;
            // submits the request to present an image to the swap chain
            result = vkQueuePresentKHR(presentQueue, &presentInfo);

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
                framebufferResized = false;
                recreateSwapChain();
            } else if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to present swap chain image!");
            }

            /* The application is rapidly submitting work in the drawFrame function, 
            but doesn't actually check if any of it finishes. If the CPU is submitting
            work faster than the GPU can keep up with then the queue will slowly fill up with work, 
            and the memory usage of the application will slowly growing */
            vkQueueWaitIdle(presentQueue);
            // with the modulo operator we ensure that the frame index loops around after every MAX_FRAMES_IN_FLIGHT enqueued frames.
            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        }

        //This function will generate a new transformation every frame to make the geometry spin around
        void updateUniformBuffer(uint32_t currentImage) {
            static auto startTime = std::chrono::high_resolution_clock::now();

            auto currentTime = std::chrono::high_resolution_clock::now();
            float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

            UniformBufferObject ubo = {};
            ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
            ubo.proj[1][1] *= -1;

            void* data;
            vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
                memcpy(data, &ubo, sizeof(ubo));
            vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
        }

        void createCommandBuffers() {
            commandBuffers.resize(swapChainFramebuffers.size());
            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = commandPool;
            //PRIMARY: Can be submitted to a queue for execution, but cannot be called from other command buffers
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

            if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate command buffers!");
            }

            // recording a command buffer
            for (size_t i = 0; i < commandBuffers.size(); i++) {
                VkCommandBufferBeginInfo beginInfo = {};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                /* we're only going to use the command buffer once and wait with returning from the 
                function until the copy operation has finished executing */
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

                if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
                    throw std::runtime_error("failed to begin recording command buffer!");
                }

                //beginning the render pass with vkCmdBeginRenderPass
                VkRenderPassBeginInfo renderPassInfo = {};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = swapChainFramebuffers[i];
                renderPassInfo.renderArea.offset = {0, 0};
                renderPassInfo.renderArea.extent = swapChainExtent;
                VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
                renderPassInfo.clearValueCount = 1;
                renderPassInfo.pClearValues = &clearColor;
                /*
                - command buffer to record the command to
                - The details of the render pass we've just provided
                - The render pass commands will be embedded in the primary command buffer itself 
                  and no secondary command buffers will be executed
                */
                vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
                VkBuffer vertexBuffers[] = {vertexBuffer};
                VkDeviceSize offsets[] = {0};
                //binding the vertex buffer during rendering operations
                vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
                vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);
                // vertexCount, instanceCount, firstVertex(offset into the vertex buffer), firstInstance( offset for instanced rendering)
                vkCmdDraw(commandBuffers[i], static_cast<uint32_t>(vertices.size()), 1, 0, 0);
                /* 
                - Descriptor sets are not unique to graphics pipelines. Therefore we need to specify if we 
                want to bind descriptor sets to the graphics or compute pipeline
                - layout that the descriptors are based on
                - index of the first descriptor set
                - number of sets to bind
                - array of sets to bind
                */
                vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
                vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
                vkCmdEndRenderPass(commandBuffers[i]);

                if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to record command buffer!");
                }

            }

        }

        void createCommandPool() {

            QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

            //Command buffers are executed by submitting them on one of the device queues
            VkCommandPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            // We're going to record commands for drawing, which is why we've chosen the graphics queue family.
            poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
            poolInfo.flags = 0; // Optional

            if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
                throw std::runtime_error("failed to create command pool!");
            }

        }

        void createFramebuffers() {
            swapChainFramebuffers.resize(swapChainImageViews.size());
            //iterate through the image views and create framebuffers from them
            for (size_t i = 0; i < swapChainImageViews.size(); i++) {
                VkImageView attachments[] = {
                    swapChainImageViews[i]
                };

                VkFramebufferCreateInfo framebufferInfo = {};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                //specify with which renderPass the framebuffer needs to be compatible
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = attachments;
                framebufferInfo.width = swapChainExtent.width;
                framebufferInfo.height = swapChainExtent.height;
                framebufferInfo.layers = 1;

                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create framebuffer!");
                }
            }
        }

/*      We need to tell Vulkan about the framebuffer attachments that will be 
        used while rendering. We need to specify how many color and depth buffers 
        there will be, how many samples to use for each of them and how their 
        contents should be handled throughout the rendering operations */
        void createRenderPass() {
            VkAttachmentDescription colorAttachment = {};
            //a single color buffer attachment represented by one of the images from the swap chain
            colorAttachment.format = swapChainImageFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            //determine what to do with the data in the attachment before rendering and after rendering.
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; //Clear the values to a constant at the start
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; //Rendered contents will be stored in memory and can be read later
            //Our application won't do anything with the stencil buffer, so the results of loading and storing are irrelevant
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            //The initialLayout specifies which layout the image will have before the render pass begins
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            //The finalLayout specifies the layout to automatically transition to when the render pass finishes
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        
            /*  
            A single render pass can consist of multiple subpasses. Subpasses are subsequent 
            rendering operations that depend on the contents of framebuffers in previous passes, 
            for example a sequence of post-processing effects that are applied one after another */
            VkAttachmentReference colorAttachmentRef = {};
            //which attachment to reference by its index in the attachment descriptions array.
            colorAttachmentRef.attachment = 0;
            //which layout we would like the attachment to have during a subpass that uses this reference.
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            //subpass definition
            VkSubpassDescription subpass = {};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            //reference to the color attachment:
            subpass.colorAttachmentCount = 1;
            //The index of the attachment in this array is directly referenced from the fragment 
            //shader with the layout(location = 0) out vec4 outColor directive!
            subpass.pColorAttachments = &colorAttachmentRef;

/*          The subpasses in a render pass automatically take care of image layout transitions. 
            These transitions are controlled by subpass dependencies, which specify memory and 
            execution dependencies between subpasses. */
            VkSubpassDependency dependency = {};
            // Indices of the dependency and the dependent subpass
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            // The operations to wait on and the stages in which these operations occur.
            // We need to wait for the swap chain to finish reading from the image before we can access it
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = 0;
/*          The operations that should wait on this are in the color attachment stage 
            and involve the reading and writing of the color attachment */
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = 1;
            renderPassInfo.pDependencies = &dependency;

            if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
                throw std::runtime_error("failed to create render pass!");
            }
        }

        void createGraphicsPipeline() {
            auto vertShaderCode = readFile("shaders/vert.spv");
            auto fragShaderCode = readFile("shaders/frag.spv");

            /* Shader modules are just a thin wrapper around the shader bytecode that we've previously 
            loaded from a file and the functions defined in it */
            VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
            VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

            //To actually use the shaders we'll need to assign them to a specific pipeline stage
            VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
            vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            //tell Vulkan in which pipeline stage the shader is going to be used. 
            vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
            //the shader module containing the code
            vertShaderStageInfo.module = vertShaderModule;
            //the function to invoke, known as the entrypoint
            vertShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
            fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            // set up the graphics pipeline to accept vertex data 
            auto bindingDescription = Vertex::getBindingDescription();
            auto attributeDescriptions = Vertex::getAttributeDescriptions();

            //structure to the format of the vertex data that will be passed to the vertex shader
            VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Optional
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());;
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // Optional

            //what kind of geometry will be drawn from the vertices and if primitive restart should be enabled
            VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            //the region of the framebuffer that the output will be rendered to. 
            VkViewport viewport = {};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            //the size of the swap chain and its images may differ from the WIDTH and HEIGHT of the window.
            viewport.width = (float) swapChainExtent.width;
            viewport.height = (float) swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            /* While viewports define the transformation from the image to the framebuffer, 
            scissor rectangles define in which regions pixels will actually be stored. 
            Any pixels outside the scissor rectangles will be discarded by the rasterizer */
            VkRect2D scissor = {};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent; //we want to draw to the entire framebuffer

            // viewport and scissor rectangle need to be combined into a viewport state
            VkPipelineViewportStateCreateInfo viewportState = {};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;

/*          The rasterizer takes the geometry that is shaped by the vertices from the vertex 
            shader and turns it into fragments to be colored by the fragment shader. */
/*          It also performs depth testing, face culling and the scissor test, and it can be configured 
            to output fragments that fill entire polygons or just the edges (wireframe rendering). */
            VkPipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.depthClampEnable = VK_FALSE; //fragments that are beyond the near and far planes are discarded
            rasterizer.rasterizerDiscardEnable = VK_FALSE; //the geometry is passed through the rasterizer stage
            // how fragments are generated for geometry
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL; //fill the area of the polygon with fragments - if VK_POLYGON_MODE_LINE, we render wireframes
            rasterizer.lineWidth = 1.0f;
            // type of face culling to use - checks all the faces that are front facing towards
            // the viewer and renders those while discarding all the faces
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            //the vertex order for faces to be considered front-facing and can be clockwise or counterclockwise
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f; // Optional
            rasterizer.depthBiasClamp = 0.0f; // Optional
            rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

/*             struct configuring multisampling, which is one of the ways to perform anti-aliasing. 
            It works by combining the fragment shader results of multiple polygons that rasterize
            to the same pixel. This mainly occurs along edges, which is also where the most noticeable 
            aliasing artifacts occur. Because it doesn't need to run the fragment shader multiple 
            times if only one polygon maps to a pixel, it is significantly less expensive than simply 
            rendering to a higher resolution and then downscaling. */
            VkPipelineMultisampleStateCreateInfo multisampling = {};
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampling.minSampleShading = 1.0f; // Optional
            multisampling.pSampleMask = nullptr; // Optional
            multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
            multisampling.alphaToOneEnable = VK_FALSE; // Optional

            /* After a fragment shader has returned a color, it needs to be combined with 
            the color that is already in the framebuffer. */
/*             The most common way to use color blending is to implement alpha blending, 
            where we want the new color to be blended with the old color based on its opacity */

            /* this settings are implementing alpha blending
            finalColor.rgb = newAlpha * newColor + (1 - newAlpha) * oldColor;
            finalColor.a = newAlpha.a; */
            VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_TRUE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

            // set blend constants that you can use as blend factors in the color blending
            // calculations specified above
            VkPipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;
            colorBlending.blendConstants[0] = 0.0f; // Optional
            colorBlending.blendConstants[1] = 0.0f; // Optional
            colorBlending.blendConstants[2] = 0.0f; // Optional
            colorBlending.blendConstants[3] = 0.0f; // Optional

/*             A limited amount of the state that we've specified in the previous 
            structs can actually be changed without recreating the pipeline. 
            Examples are the size of the viewport, line width and blend constants. To do that
            we need this structure */
            VkDynamicState dynamicStates[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_LINE_WIDTH
            };

            VkPipelineDynamicStateCreateInfo dynamicState = {};
            dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

/*          The structure also specifies push constants, 
            which are another way of passing dynamic values to shaders */
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1; 
            pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

            if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
                throw std::runtime_error("failed to create pipeline layout!");
            }

            /* 
            Combining:
            -  The shader modules that define the functionality of the programmable stages of the graphics pipeline
            -  All of the structures that define the fixed-function stages of the pipeline, like input assembly, rasterizer, viewport and color blending
            -  The uniform and push values referenced by the shader that can be updated at draw time
            -  The attachments referenced by the pipeline stages and their usage
            */
            VkGraphicsPipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = nullptr; // Optional
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = nullptr; // Optional

            pipelineInfo.layout = pipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
            pipelineInfo.basePipelineIndex = -1; // Optional

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
                throw std::runtime_error("failed to create graphics pipeline!");
            }

/*          The compilation and linking of the SPIR-V bytecode to machine code for execution by the GPU
            doesn't happen until the graphics pipeline is created. 
            So we're allowed to destroy the shader modules again as soon as pipeline creation is finished, */
            vkDestroyShaderModule(device, fragShaderModule, nullptr);
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
            
        }
        //we have to wrap the shader code in a VkShaderModule
        VkShaderModule createShaderModule(const std::vector<char>& code) {
            VkShaderModuleCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

            VkShaderModule shaderModule;
            if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
                throw std::runtime_error("failed to create shader module!");
            }
            return shaderModule; //return the wrapper around the shader bytecode
        }

        void createImageViews() {
            // resize the list to fit all of the image views we'll be creating
            swapChainImageViews.resize(swapChainImages.size());
            for (size_t i = 0; i < swapChainImages.size(); i++) {
                VkImageViewCreateInfo createInfo = {};
                createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                createInfo.image = swapChainImages[i];
                //how the image data should be interpreted
                createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                createInfo.format = swapChainImageFormat;
                //swizzle the color channels around. Identity is default mapping
                createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
                //What the image's purpose is and which part of the image should be accessed
                createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;
                if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create image views!");
                }
            }


        }

        void cleanupSwapChain() {

            for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
                vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
            }

            vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyRenderPass(device, renderPass, nullptr);

            for (size_t i = 0; i < swapChainImageViews.size(); i++) {
                vkDestroyImageView(device, swapChainImageViews[i], nullptr);
            }

            vkDestroySwapchainKHR(device, swapChain, nullptr);
            
            /* 
            Since the uniform buffer also depends on the number of swap chain 
            images, which could change after a recreation, we clean it up here
            */
            for (size_t i = 0; i < swapChainImages.size(); i++) {
                vkDestroyBuffer(device, uniformBuffers[i], nullptr);
                vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
            }
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        }

        /* Vulkan will usually just tell us that the swap chain is no longer adequate during presentation
           In that case we will need to call this function 
        */
        void recreateSwapChain() {
            //special kind of window resizing: when framebuffer size is 0
            int width = 0, height = 0;
            while (width == 0 || height == 0) {
                glfwGetFramebufferSize(window, &width, &height);
                glfwWaitEvents();
            }

            vkDeviceWaitIdle(device); //we shouldn't touch resources that may still be in use

            //make sure that the old versions of these objects are cleaned up before recreating them
            cleanupSwapChain();

            createSwapChain();
            createImageViews();
            createRenderPass();
            createGraphicsPipeline();
            createFramebuffers();
            createUniformBuffers();
            createDescriptorPool();
            createDescriptorSets();
            createCommandBuffers();
        }

        void createDescriptorSets() {

            /*  
            specify the descriptor pool to allocate from, the number of descriptor sets to allocate, 
            and the descriptor layout to base them on
            */
            //we will create one descriptor set for each swap chain image, all with the same layout. 
            std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
            VkDescriptorSetAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptorPool;
            allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
            allocInfo.pSetLayouts = layouts.data();

            descriptorSets.resize(swapChainImages.size());
            
            //You don't need to explicitly clean up descriptor sets, because they will be automatically 
            //freed when the descriptor pool is destroyed
            if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate descriptor sets!");
            }
            /* The descriptor sets have been allocated now, but the descriptors 
            within still need to be configured*/

            for (size_t i = 0; i < swapChainImages.size(); i++) {
                VkDescriptorBufferInfo bufferInfo = {};
                bufferInfo.buffer = uniformBuffers[i];
                bufferInfo.offset = 0;
                bufferInfo.range = sizeof(UniformBufferObject);

                VkWriteDescriptorSet descriptorWrite = {};

                descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                //The first two fields specify the descriptor set to update and the binding
                descriptorWrite.dstSet = descriptorSets[i];
                descriptorWrite.dstBinding = 0;
                /* We need to specify the type of descriptor again. It's possible 
                to update multiple descriptors at once in an array, starting at 
                index dstArrayElement. The descriptorCount field specifies how 
                many array elements you want to update. */
                descriptorWrite.dstArrayElement = 0;

                descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                //It depends on the type of descriptor which one of the three you actually need to use
                descriptorWrite.descriptorCount = 1;
                //is used for descriptors that refer to buffer data
                descriptorWrite.pBufferInfo = &bufferInfo;

                vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);


            }

        }

        void createSwapChain() {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

            VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
            VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
            VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
            //how many images we would like to have in the swap chain 
            uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
            // make sure to not exceed the maximum number of images 
            if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
                imageCount = swapChainSupport.capabilities.maxImageCount;
            }

            VkSwapchainCreateInfoKHR createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            createInfo.surface = surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
            uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

            if (indices.graphicsFamily != indices.presentFamily) {
                createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices = queueFamilyIndices;
            } else {
                createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                createInfo.queueFamilyIndexCount = 0; // Optional
                createInfo.pQueueFamilyIndices = nullptr; // Optional
            }

            createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            createInfo.presentMode = presentMode;
            createInfo.clipped = VK_TRUE;
            createInfo.oldSwapchain = VK_NULL_HANDLE;

            if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
                throw std::runtime_error("failed to create swap chain!");
            }

            // retrieve the handles
            vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
            swapChainImages.resize(imageCount);
            vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

            swapChainImageFormat = surfaceFormat.format;
            swapChainExtent = extent;
        }

        void createSurface() {
            if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
                throw std::runtime_error("failed to create window surface!");
            }
        }

        void createLogicalDevice() {

            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
            //we need to have multiple VkDeviceQueueCreateInfo structs to create a queue from both families.
            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
            std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

            //priorities to queues to influence the scheduling of command buffer execution
            float queuePriority = 1.0f;
            for (uint32_t queueFamily : uniqueQueueFamilies) {
                VkDeviceQueueCreateInfo queueCreateInfo = {};
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.queueFamilyIndex = queueFamily;
                queueCreateInfo.queueCount = 1;
                queueCreateInfo.pQueuePriorities = &queuePriority;
                queueCreateInfos.push_back(queueCreateInfo);
            }

            //the set of device features that we'll be using
            VkPhysicalDeviceFeatures deviceFeatures = {};

            VkDeviceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

            createInfo.pQueueCreateInfos = queueCreateInfos.data();
            createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

            createInfo.pEnabledFeatures = &deviceFeatures;

            createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
            createInfo.ppEnabledExtensionNames = deviceExtensions.data();
            
            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            } else {
                createInfo.enabledLayerCount = 0;
            }

            if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
                throw std::runtime_error("failed to create logical device!");
            }

            vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
            vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
        }

        void pickPhysicalDevice() {
            uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
            if (deviceCount == 0) {
                throw std::runtime_error("failed to find GPUs with Vulkan support!");
            }

            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

            for (const auto& device : devices) {
                if (isDeviceSuitable(device)) {
                    physicalDevice = device;
                    break;
                }
            }

            if (physicalDevice == VK_NULL_HANDLE) {
                throw std::runtime_error("failed to find a suitable GPU!");
            }

        }
        //evaluate the suitability of a device
        bool isDeviceSuitable(VkPhysicalDevice device) {
            QueueFamilyIndices indices = findQueueFamilies(device);
            bool extensionsSupported = checkDeviceExtensionSupport(device);
            bool swapChainAdequate = false;
            if (extensionsSupported) {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
                swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            }
            return indices.isComplete() && extensionsSupported && swapChainAdequate;
        }
        //enumerate the extensions and check if all of the required extensions are amongst them.
        bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

            std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

            for (const auto& extension : availableExtensions) {
                requiredExtensions.erase(extension.extensionName);
            }

            return requiredExtensions.empty();
        }

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
            SwapChainSupportDetails details;
            //determining the supported capabilitie
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
            // querying the supported surface formats
            uint32_t formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

            if (formatCount != 0) {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
            }
            // querying the supported presentation modes
            uint32_t presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
            }

            return details;
        }

        //every operation in Vulkan, anything from drawing to uploading textures, 
        //requires commands to be submitted to a queue.
        //We need to check which queue families are supported by the device and which
        //one of these supports the commands that we want to use. 
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
            QueueFamilyIndices indices;

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto& queueFamily : queueFamilies) {
                if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    indices.graphicsFamily = i;
                }
                // look for a queue family that has the capability of presenting to our window surface
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

                if (queueFamily.queueCount > 0 && presentSupport) {
                    indices.presentFamily = i;
                }

                if (indices.isComplete()) {
                    break;
                }

                i++;
            }

            return indices;
        }

        void createInstance() {
            if (enableValidationLayers && !checkValidationLayerSupport()) {
                throw std::runtime_error("validation layers requested, but not available!");
            }

            VkApplicationInfo appInfo = {};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "Hello Triangle";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;

            VkInstanceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;

            auto extensions = getRequiredExtensions();
            createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            createInfo.ppEnabledExtensionNames = extensions.data();

            VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();

                populateDebugMessengerCreateInfo(debugCreateInfo);
                createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
            } else {
                createInfo.enabledLayerCount = 0;
                
                createInfo.pNext = nullptr;
            }

            if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
                throw std::runtime_error("failed to create instance!");
            }

        }

        void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
            createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            createInfo.pfnUserCallback = debugCallback;
        }

        void setupDebugMessenger() {
            if (!enableValidationLayers) return;

            VkDebugUtilsMessengerCreateInfoEXT createInfo;
            populateDebugMessengerCreateInfo(createInfo);

            if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
                throw std::runtime_error("failed to set up debug messenger!");
            }
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

        // Each VkSurfaceFormatKHR entry contains a format and a colorSpace member. 
        // The format member specifies the color channels and types.
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

            for (const auto& availableFormat : availableFormats) {
                if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return availableFormat;
                }
            }

            return availableFormats[0];
        }

        // The presentation mode is arguably the most important setting for the swap chain, 
        // because it represents the actual conditions for showing images to the scree
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
            VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

            for (const auto& availablePresentMode : availablePresentModes) {
                if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    return availablePresentMode;
                } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                    bestMode = availablePresentMode;
                }
            }

            return bestMode;
        }
        // The swap extent is the resolution of the swap chain images and it's 
        // almost always exactly equal to the resolution of the window that we're drawing to
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            } else {
/*              Handle window resizes properly, we also need to query the current size of the framebuffer 
                to make sure that the swap chain images have the (new) right size.  */
                int width, height;
                glfwGetFramebufferSize(window, &width, &height);

                VkExtent2D actualExtent = {
                            static_cast<uint32_t>(width),
                            static_cast<uint32_t>(height)
                        };

                actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
                actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

                return actualExtent;
            }
        }

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
};


#endif