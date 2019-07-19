#ifndef EDITOR_H
#define EDITOR_H

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS

/*
the problem of ordering fragments by depth is much more commonly solved using a
depth buffer.
A depth buffer is an additional attachment that stores the depth for every
position.
Every time the rasterizer produces a fragment, the depth test will
check if the new fragment is closer than the previous one
 */
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define WIDTH 800
#define HEIGHT 600

#include "GLFW/glfw3.h"
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
#include <unordered_map>
#include <sstream>
#include <math.h>
#include <yocto_math.h>

inline const double pi = 3.14159265358979323846;

struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec3 color;
  glm::vec2 texCoord;

  // A vertex binding describes at which rate to load data from memory
  // throughout the vertices
  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  // An attribute description struct describes how to extract a vertex attribute
  // from a chunk of vertex data originating from a binding description
  static std::array<VkVertexInputAttributeDescription, 4>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions = {};
    // position attribute
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);
    // normal attribute
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);
    // color attribute
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, color);
    // texcoord attribute
    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Vertex, texCoord);

    return attributeDescriptions;
  }

  bool operator==(const Vertex &other) const {
    return pos == other.pos && color == other.color &&
           texCoord == other.texCoord;
  }
};

struct Texture {
  int texWidth;
  int texHeight;
  int texChannels;
  unsigned char *pixels;
};

struct Model {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  std::vector<glm::vec3> faces;
  Texture txt;
  glm::vec3 bbox_min;
  glm::vec3 bbox_max;
};

namespace std {
template <> struct hash<Vertex> {
  size_t operator()(Vertex const &vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.color) << 1)) >>
            1) ^
           (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};
}

struct Camera {
  std::string uri = "";
  yocto::frame3f frame = yocto::identity3x4f;
  bool orthographic = false;
  float lens = 0;
  yocto::vec2f film = {0.036, 0.015};
  float focus = yocto::flt_max;
  float aperture = 0;
};

struct Light {
  glm::vec3 position;
  glm::vec3 color;
  float intensity;
};

// descriptor
struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
  alignas(4) glm::vec2 resolution;
  alignas(16) glm::vec3 light_position;
  alignas(16) glm::vec3 light_color;
  alignas(4) float light_intensity;
};

/* The stages that the current frame has already progressed through are idle
and could already be used for a next frame*/
const int MAX_FRAMES_IN_FLIGHT = 2;

// validation layers are optional components that hook into Vulkan function
// calls to apply additional operations.
// Once defined , they currently have no way to relay the debug messages back to
// our program.
// To receive those messages we have to set up a debug messenger with a
// callback,
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

/* Vulkan does not have the concept of a "default framebuffer", hence it
requires an infrastructure
that will own the buffers we will render to before we visualize them on the
screen.
This infrastructure is known as the swap chain and must be created explicitly in
Vulkan.
The swap chain is essentially a queue of images that are waiting to be presented
to the screen.

image representation is tied to the windows system, so it is not part of the
Vulkan Core.
We need to enable enable the VK_KHR_swapchain device extension after querying
for its support.
*/
const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

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

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

static std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();
  return buffer;
}

class Viewer {
public:
  Model model;
  Camera camera;
  Light light;
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanUp();
  }

private:
  GLFWwindow *window;
  // The instance is the connection between your application and the Vulkan
  // library
  VkInstance instance;
  VkDebugUtilsMessengerEXT
      debugMessenger; // tell Vulkan about the callback function
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  // After selecting a physical device to use we
  // need to set up a logical device to interface with it.
  VkDevice device;
  // queue with graphics capabilities
  VkQueue graphicsQueue;
  // create the presentation queue
  VkQueue presentQueue;
  // Vulkan is a platform agnostic API, it can not interface directly
  // with the window system on its own.To establish the connection between
  // Vulkan
  // and the window system to present results to the screen,
  // we need to use the WSI (Window System Integration) extensions.
  VkSurfaceKHR surface;
  VkSwapchainKHR swapChain;
  // retrieving the handles of the VkImages in the swap chain
  std::vector<VkImage> swapChainImages;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  // To use any VkImage, including those in the swap chain, in the render
  // pipeline we have to create
  // a VkImageView object. An image view is quite literally a view into an
  // image.
  // It describes how to access the image and which part of the image to access
  std::vector<VkImageView> swapChainImageViews;

  // All of the descriptor bindings are combined into this object
  VkDescriptorSetLayout descriptorSetLayout;
  /* if we want to change the behaviour of the shader at drawing time without
  recreating it, we can user
  uniform values in the shader. Those uniforms are used to pass the
  transformation matrix
  in the vertex shader, or to create texture samples in the fragment shader.
  These uniforms needs to be specified during the pipeline creation by creating
  a pipeline layout struct */
  VkPipelineLayout pipelineLayout;
  VkRenderPass renderPass;

  VkPipeline graphicsPipeline;

  /*
  - The attachments specified during render pass creation are bound by wrapping
  them into a VkFramebuffer object
  - A framebuffer object references all of the VkImageView objects that
  represent the attachments
  - the image that we have to use for the attachment depends on which image the
  swap chain returns
  when we retrieve one for presentation
  - we have to create a framebuffer for all of the images in the swap chain and
  use the one that
  corresponds to the retrieved image at drawing time.
  */
  std::vector<VkFramebuffer> swapChainFramebuffers;

  // we have to record all of the operations you want to perform in command
  // buffer objects
  VkCommandPool commandPool;
  /*      Because one of the drawing commands involves binding the right
     VkFramebuffer,
          we'll actually have to record a command buffer for every image in the
     swap chain once again.  */
  std::vector<VkCommandBuffer> commandBuffers;

  // Each frame should have its own set of semaphores
  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  // To use the right pair of semaphores every time, we need to keep track of
  // the current frame
  size_t currentFrame = 0;

  /*       To perform CPU-GPU synchronization, we build a fence for each frame
          Fences are mainly designed to synchronize your application itself with
     rendering operation,
          whereas semaphores are used to synchronize operations within or across
     command queues.  */
  std::vector<VkFence> inFlightFences;
  // member variable that flags that a resize has happened
  bool framebufferResized = false;

  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;
  VkBuffer indexBuffer;
  VkDeviceMemory indexBufferMemory;

  // buffer that contains the UBO data for the shader

  /*
  copy new data to the uniform buffer every frame, so it doesn't really make
  any sense to have a staging buffer. It would just add extra overhead in this
  case and likely degrade performance instead of improving it.
   */
  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformBuffersMemory;

  VkDescriptorPool descriptorPool;
  // hold the descriptor set handles
  std::vector<VkDescriptorSet> descriptorSets;

  // We only need a single depth image, because only one draw operation is
  // running at once
  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;

  VkImage textureImage;
  VkDeviceMemory textureImageMemory;
  /* images are accessed through image views rather than directly
   we need to create such an image view for the texture image too */
  VkImageView textureImageView;

  VkSampler textureSampler;

  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    // using this glfw function we set inside THIS an arbitray pointer to a
    // GLFWwindow
    // so inside the framebufferResizeCallback i can reference to GLFWwindow*
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
    // must be created after logical device creation
    createSwapChain();
    createImageViews();
    createRenderPass();
    // We need to provide details about every descriptor binding used in the
    // shaders for pipeline creation
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    // buffers do not automatically allocate memory for themselves. We must do
    // that by our own
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }
  void mainLoop() {
    auto mouse_pos = yocto::zero2f, last_pos = yocto::zero2f;
    double lastTime = glfwGetTime();
    int nbFrames = 0;
    while (!glfwWindowShouldClose(window)) {
      double currentTime = glfwGetTime();
      double delta = currentTime - lastTime;
      nbFrames++;
      if (delta >= 1.0) {

        double fps = double(nbFrames) / delta;

        std::stringstream ss;
        ss << " [" << fps << " FPS]";

        glfwSetWindowTitle(window, ss.str().c_str());

        nbFrames = 0;
        lastTime = currentTime;
      }
      last_pos = mouse_pos;
      mouse_pos = get_mouse_position(window);
      int mouse_left = is_mouse_left(window);
      int mouse_right = is_mouse_right(window);
      auto alt_down = is_alt_key(window);
      auto shift_down = is_shift_key(window);

      if ((mouse_left || mouse_right) && !alt_down) {
        auto dolly = 0.0f;
        auto pan = yocto::zero2f;
        auto rotate = yocto::zero2f;
        if (mouse_left && !shift_down) {
          rotate = (mouse_pos - last_pos) / 100.0f;
        }
        if (mouse_right)
          dolly = (mouse_pos.x - last_pos.x) / 100.0f;
        if (mouse_left && shift_down)
          pan = (mouse_pos - last_pos) / 100.0f;
        update_turntable(camera.frame, camera.focus, rotate, dolly, pan);
      }

      updateLight(light);
      glfwPollEvents();
      drawFrame();
    }
    // wait for the logical device to finish operations before exiting and
    // destroy the window
    vkDeviceWaitIdle(device);
  }

  void cleanUp() {
    cleanupSwapChain();

    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);

    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

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

  /*
  Textures are usually accessed through samplers, which will apply filtering
  and transformations to compute the final color that is retrieved.
  These filters are helpful to deal with problems like oversampling:

  - a texture that is mapped to geometry with more fragments than texels, so
    you simply took the closest texel for the texture coordinate in each
  fragment, and
    you will have a minecraft style result

  Oversampling is the opposite problem: you have more texels than fragments
  This will lead to artifacts when sampling high frequency patterns (like
  checkerboard texture)

  sampler can also take care of transformations.  It determines what happens
  when you try
  to read texels outside the image through its addressing mode

  */
  void createTextureSampler() {
    // structure which specifies all filters and transformations that it should
    // apply.
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // how to interpolate texels that are magnified or minified
    /*
    Magnification concerns the oversampling problem describes above,
    and minification concerns undersampling
     */
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;

    samplerInfo.addressModeU =
        VK_SAMPLER_ADDRESS_MODE_REPEAT; // Repeat the texture when going beyond
                                        // the image dimensions
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    samplerInfo.anisotropyEnable = VK_TRUE; // use anisotropic filtering
    /* limits the amount of texel samples that can be used to calculate the
    final color.
    A lower value results in better performance, but lower quality results */
    samplerInfo.maxAnisotropy = 16;
    // which color is returned when sampling beyond the image with clamp to
    // border addressing mode
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    // which coordinate system you want to use to address texels in an image
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    /*
    If a comparison function is enabled, then texels will first be compared to a
    value,
    and the result of that comparison is used in filtering operations.
    This is mainly used for percentage-closer filtering on shadow maps
     */
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    // mipmapping
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create texture sampler!");
    }
  }

  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    // how the image data should be interpreted
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    // What the image's purpose is and which part of the image should be
    // accessed
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
  }

  void createTextureImageView() {

    textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                       VK_IMAGE_ASPECT_COLOR_BIT);
  }

  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                         uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    // specify which part of the buffer is going to be copied to which part of
    // the image
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(
        commandBuffer, buffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, // which layout the image is
                                              // currently using
        1, &region);

    endSingleTimeCommands(commandBuffer);
  }

  /*
  If we were still using buffers, then we could now write a function to record
  and execute
  vkCmdCopyBufferToImage to finish the job, but this command requires the image
  to be in the
  right layout first. We create this function to handle layout transitions
  */
  void transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    /*
    The most common way to performlayout transitions is using an image memory
    barrier.
    A pipeline barrier like that is generally used to synchronize access to
    resources.
    */
    VkImageMemoryBarrier barrier = {};
    // fields for specify layout transition
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    // If you are using the barrier to transfer queue family ownership, then
    // these two fields should be the indices of the queue families.
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    barrier.image = image; // image that is affected

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

      if (hasStencilComponent(format)) {
        barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      }
    } else {
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    // the specific part of the image which is affected
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    /*
    Barriers are primarily used for synchronization purposes, so you must
    specify which types
    of operations that involve the resource must happen before the barrier, and
    which operations
    that involve the resource must wait on the barrier. We need to do that
    despite already using
    vkQueueWaitIdle to manually synchronize
    */
    /*             barrier.srcAccessMask = 0; // TODO
                barrier.dstAccessMask = 0; // TODO */

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    /*
    we need to manage two transitions:
    - transfer writes that don't need to wait on anything
    - shader reads should wait on transfer writes, specifically the shader reads
    in the fragment shader,
      because that's where we're going to use the texture
    */
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
               newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, // in which pipeline stage the operations occur that should
                     // happen before the barrier
        destinationStage, // The second parameter specifies the pipeline stage
                          // in which operations will wait on the barrier
        0, // 0 or VK_DEPENDENCY_BY_REGION_BIT. The latter turns the barrier
           // into a per-region condition. That means that the implementation is
           // allowed to already begin reading from the parts of a resource that
           // were written so far
        0, nullptr, // reference arrays of memory barriers
        0, nullptr, // reference arrays of buffer memory barriers
        1, &barrier // reference arrays of image memory barriers
        );

    endSingleTimeCommands(commandBuffer);
  }

  void createImage(uint32_t width, uint32_t height, VkFormat format,
                   VkImageTiling tiling, VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties, VkImage &image,
                   VkDeviceMemory &imageMemory) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling; // Texels are laid out in an implementation
                               // defined order for optimal access
    imageInfo.initialLayout =
        VK_IMAGE_LAYOUT_UNDEFINED; // Not usable by the GPU and the very first
                                   // transition will discard the texels
    // The image is going to be used as destination for the buffer copy, so it
    // should be set up as a transfer destination
    // We also want to be able to access the image from the shader to color our
    // mesh
    imageInfo.usage = usage;
    // The image will only be used by one queue family: the one that supports
    // graphics
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // The samples flag is related to multisampling. This is only relevant for
    // images that will be used as attachments
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    // If you were using a 3D texture for a voxel terrain, for example, then you
    // could use this to avoid allocating memory to store large volumes of "air"
    // values
    imageInfo.flags = 0; // Optional

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
  }

  void createTextureImage() {

    VkDeviceSize imageSize = model.txt.texWidth * model.txt.texHeight * 4;

    if (!model.txt.pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    // copy the pixel values that we got from the image loading library to the
    // buffer
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, model.txt.pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);

    createImage(model.txt.texWidth, model.txt.texHeight,
                VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage,
                textureImageMemory);

    // Transition the texture image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // Execute the buffer to image copy operation
    copyBufferToImage(stagingBuffer, textureImage,
                      static_cast<uint32_t>(model.txt.texWidth),
                      static_cast<uint32_t>(model.txt.texHeight));
    // to start sampling from the texture image in the shader, we need one last
    // transition
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT;
  }

  // The support of a format depends on the tiling mode and usage, so we must
  // also include these as parameters.
  VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
      if (tiling == VK_IMAGE_TILING_LINEAR &&
          (props.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                 (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  // select a format with a depth component that supports usage as depth
  // attachment
  VkFormat findDepthFormat() {
    return findSupportedFormat({VK_FORMAT_D32_SFLOAT,
                                VK_FORMAT_D32_SFLOAT_S8_UINT,
                                VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  void createDepthResources() {
    /*
    A depth image should have the same resolution as the color attachment,
    defined by the swap chain extent, an image usage appropriate for a depth
    attachment,
    optimal tiling and device local memory.
    We don't necessarily need a specific format, because we won't be directly
    accessing
    the texels from the program. It just needs to have a reasonable accuracy,
    at least 24 bits is common in real-world applications.
    Along with accuracy for depth test, we need also come accuracy for stencil
    test,
    which is an additional test that can be combined with depth testing. We use
    findSupportedFormat function to search for the proper format.
    */

    VkFormat depthFormat = findDepthFormat();

    createImage(
        swapChainExtent.width, swapChainExtent.height, depthFormat,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);

    depthImageView =
        createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    // the depth image need to be transitioned to a layout that is suitable for
    // depth attachment usage
    // we use a pipeline barrier because the transition only needs to happen
    // once
    transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  // light update
  void updateLight(Light &light) {

    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    float distance = sqrt(light.position.x * light.position.x +
                          light.position.y * light.position.y +
                          light.position.z * light.position.z);

    light.position.x = distance * cos(time / 100.0f * 90.0f);
    light.position.y = distance * sin(time / 100.0f * 90.0f);
  }

  // MOUSE MOVEMENT
  int is_mouse_left(GLFWwindow *window) {
    return glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  }
  int is_mouse_right(GLFWwindow *window) {
    return glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  }
  int is_mouse_middle(GLFWwindow *window) {
    return glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_3) == GLFW_PRESS;
  }

  bool is_alt_key(GLFWwindow *window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
  }

  bool is_shift_key(GLFWwindow *window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  }

  yocto::vec2f get_mouse_position(GLFWwindow *window) {
    double mouse_posx, mouse_posy;
    glfwGetCursorPos(window, &mouse_posx, &mouse_posy);
    auto pos = yocto::vec2f{(float)mouse_posx, (float)mouse_posy};
    return pos;
  }

  // updates the uniform buffer with a new transformation every frame
  void createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(swapChainImages.size());
    uniformBuffersMemory.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   uniformBuffers[i], uniformBuffersMemory[i]);
    }
  }

  // Descriptor sets can't be created directly, they must be allocated from a
  // pool like command buffers
  void createDescriptorPool() {
    // describe which descriptor types our descriptor sets are going to contain
    // and how many of them
    std::array<VkDescriptorPoolSize, 2> poolSizes = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount =
        static_cast<uint32_t>(swapChainImages.size());
    // we include also a combined image samples descriptor in the descriptor set
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount =
        static_cast<uint32_t>(swapChainImages.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    // specify the maximum number of descriptor sets that may be allocated
    poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
  }

  // We need to provide details about every descriptor binding used in the
  // shaders for pipeline creation, as we did for
  // location index
  void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    // binding used in the shader and the type of descriptor, which is a uniform
    // buffer object
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    // in which shader stages the descriptor is going to be referenced
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    // image sampling related descriptors
    uboLayoutBinding.pImmutableSamplers = nullptr;

    /* Combined image sampler descriptor. This descriptor makes it possible for
    shaders
    to access an image resource through a sampler object */
    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags =
        VK_SHADER_STAGE_FRAGMENT_BIT; // we intend to use the combined image
                                      // sampler descriptor in the fragment
                                      // shader

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding, samplerLayoutBinding};
    // create the VkDescriptorSetLayout object for descriptor bindings
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                    &descriptorSetLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }
  }

  void createIndexBuffer() {
    VkDeviceSize bufferSize = sizeof(model.indices[0]) * model.indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, model.indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer,
                 indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    // for which purposes the data in the buffer is going to be used
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create buffer!");
    }

    // assigning memory to the buffer
    VkMemoryRequirements memRequirements;
    // query for the buffer memory requirements
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    // associate the memory with the buffer
    /*
    Since this memory is allocated specifically for this the vertex buffer, the
    offset is simply 0.
    If the offset is non-zero, then it is required to be divisible by
    memRequirements.alignment.
    */
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
  }

  /* The most optimal memory has the VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT flag
  and
  is usually not accessible by the CPU on dedicated graphics cards

  we're going to create two vertex buffers:
  - One staging buffer in CPU accessible memory to upload the data from the
  vertex array to.
  - The final vertex buffer in device local memory.
  */

  void createVertexBuffer() {
    VkDeviceSize bufferSize = sizeof(model.vertices[0]) * model.vertices.size();

    // stagingBuffer with stagingBufferMemory for mapping and copying the vertex
    // data.
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    /*
    The driver may not immediately copy the data into the buffer memory,
    for example because of caching. It is also possible that writes to the
    buffer are not visible in the mapped memory yet. To resolve that,
    use a memory heap that is host coherent, indicated with
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    */
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    // We're using a new stagingBuffer with stagingBufferMemory for mapping and
    // copying the vertex data
    void *data;
    // access a region of the specified memory resource defined by an offset and
    // size
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    // memcpy the vertex data to the mapped memory and unmap it again using
    // vkUnmapMemory
    memcpy(data, model.vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    /*
    we can copy data from the stagingBuffer to the vertexBuffer.
    We have to indicate that we intend to do that by specifying the transfer
    source flag
    for the stagingBuffer and the transfer destination flag for the
    vertexBuffer,
    along with the vertex buffer usage flag. */
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer,
                 vertexBufferMemory);
    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  VkCommandBuffer beginSingleTimeCommands() {
    // Memory transfer operations are executed using command buffers,
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    // Start recording the command buffer:
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

    /* Unlike the draw commands, there are no events we need to wait on this
    time.
    We just want to execute the transfer on the buffers immediately
    Wait for the transfer queue to become idle with vkQueueWaitIdle.
    */
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  }

  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion = {};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
  }

  /*
  Graphics cards can offer different types of memory to allocate from.
  We need to combine the requirements of the buffer and our own application
  requirements to find the right type of memory to use*/
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    // query info about the available types of memory
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    /*
    Memory heaps are distinct memory resources like dedicated VRAM and swap
    space
    in RAM for when VRAM runs out. The different types of memory exist within
    these heaps
    */
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      /*
      we can find the index of a suitable memory type by simply iterating over
      them and checking if the corresponding bit is set to

      We also need to be able to write our vertex data to that memory.
      VkMemoryType structures in
      memProperties specify heap and properties for each type of memory. One of
      the property is if
      the memory can be mapped, so we can write to it from the cpu. Since we
      need to write
      vertex buffer in this memory, we need to check for that property too.
      */
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }
    throw std::runtime_error("failed to find suitable memory type!");
  }

  /*      Detect resizes with the GLFW framework, creating a callback
          We create static function as a callback is because GLFW does not know
     how to properly call
          a member function with the right this pointer to our Viewer instance.
     */
  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    auto app = reinterpret_cast<Viewer *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    /*          By default, fences are created in the unsignaled state.
                we can change the fence creation to initialize it in the
       signaled state as
                if we had rendered an initial frame that finished */
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &imageAvailableSemaphores[i]) != VK_SUCCESS ||
          vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &renderFinishedSemaphores[i]) != VK_SUCCESS ||
          vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) !=
              VK_SUCCESS) {

        throw std::runtime_error(
            "failed to create synchronization objects for a frame!");
      }
    }
  }

  /*
  - Acquire an image from the swap chain
  - Execute the command buffer with that image as attachment in the framebuffer
  - Return the image to the swap chain for presentation

  Each of these events is set in motion using a single function call, but they
  are executed asynchronously
  We want to synchronize the queue operations of draw commands and presentation
  */
  void drawFrame() {
    // takes an array of fences and waits for either any or all of them to be
    // signaled before returning
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                    std::numeric_limits<uint64_t>::max());

    // acquire an image from the swap chain
    uint32_t imageIndex;
    /*
    vkAcquireNextImageKHR return two kind of information which can tell us if
    we need to restore the swap chain:
    - VK_ERROR_OUT_OF_DATE_KHR The swap chain has become incompatible with the
    surface and can no longer be used for rendering (when you resize window)
    - VK_SUBOPTIMAL_KHR the surface properties are no longer matched exactly.
    */
    VkResult result = vkAcquireNextImageKHR(
        device, swapChain, std::numeric_limits<uint64_t>::max(),
        imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapChain();
      return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(imageIndex);

    // Queue submission and synchronization
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // We want to wait with writing colors to the image until it's available,
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    // if we abort drawing in the PREVIOUS cycle,
    // at this point then the fence will never have been submitted with
    // vkQueueSubmit,
    // being in THIS cycle in a possible unexpected state. So we reset fences
    // here.
    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    // We can now submit the command buffer to the graphics queue
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                      inFlightFences[currentFrame]) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    /*          The last step of drawing a frame is submitting the result back
       to the swap chain to
                have it eventually show up on the screen */
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    // which semaphores to wait on before presentation can happen
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    // the swap chains to present images to and the index of the image for each
    // swap chain
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    /*          Results. It allows you to specify an array of VkResult values to
       check for every
                individual swap chain if presentation was successful. It's not
       necessary if you're only
                using a single swap chain, because you can simply use the return
       value of the present function */
    // presentInfo.pResults = nullptr;
    // submits the request to present an image to the swap chain
    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
    } else if (result != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    /* The application is rapidly submitting work in the drawFrame function,
    but doesn't actually check if any of it finishes. If the CPU is submitting
    work faster than the GPU can keep up with then the queue will slowly fill up
    with work,
    and the memory usage of the application will slowly growing */
    // vkQueueWaitIdle(presentQueue);
    // with the modulo operator we ensure that the frame index loops around
    // after every MAX_FRAMES_IN_FLIGHT enqueued frames.
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  yocto::vec2f camera_fov(const Camera &camera) {
    assert(!camera.orthographic);
    return {2 * atan(camera.film.x / (2 * camera.lens)),
            2 * atan(camera.film.y / (2 * camera.lens))};
  }

  glm::mat4 frame_to_mat(yocto::frame3f frame) {
    return glm::mat4(frame.x.x, frame.x.y, frame.x.z, 0, frame.y.x, frame.y.y,
                     frame.y.z, 0, frame.z.x, frame.z.y, frame.z.z, 0,
                     frame.o.x, frame.o.y, frame.o.z, 1);
  }

  glm::vec4 get_glframebuffer_viewport(GLFWwindow *win) {

    auto yviewport = yocto::zero4i;
    glfwGetFramebufferSize(win, &yviewport.z, &yviewport.w);
    glm::vec4 viewport(0, 0, yviewport.z, yviewport.w);
    return viewport;
  }

  inline glm::mat4 perspective_mat(float fovy, float aspect, float near,
                                   float far) {
    auto tg = tan(fovy / 2);
    return {{1 / (aspect * tg), 0, 0, 0},
            {0, 1 / tg, 0, 0},
            {0, 0, (far + near) / (near - far), -1},
            {0, 0, 2 * far * near / (near - far), 0}};
  }

  // This function will generate a new transformation every frame to make the
  // geometry spin around
  void updateUniformBuffer(uint32_t currentImage) {

    UniformBufferObject ubo = {};
    ubo.model = glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    ubo.view = glm::inverse(frame_to_mat(camera.frame));
    glm::vec4 viewport = get_glframebuffer_viewport(window);
    ubo.proj = perspective_mat(
        camera_fov(camera).x * (float)viewport.w / (float)viewport.z,
        (float)viewport.z / (float)viewport.w, 0.01f, 10000.0f);
    /*
    GLM was originally designed for OpenGL, where the Y coordinate of the clip
    coordinates is
    inverted. The easiest way to compensate for that is to flip the sign on the
    scaling factor
    of the Y axis in the projection matrix. If you don't do this, then the image
    will be rendered
    upside down
    */
    ubo.proj[1][1] *= -1;

    ubo.light_color = light.color;
    ubo.light_position = light.position;
    ubo.light_intensity = light.intensity;

    void *data;
    vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0,
                &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
  }

  void createCommandBuffers() {
    commandBuffers.resize(swapChainFramebuffers.size());
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    // PRIMARY: Can be submitted to a queue for execution, but cannot be called
    // from other command buffers
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }

    // recording a command buffer
    for (size_t i = 0; i < commandBuffers.size(); i++) {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      /* we're only going to use the command buffer once and wait with returning
      from the
      function until the copy operation has finished executing */
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

      if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
      }

      // beginning the render pass with vkCmdBeginRenderPass
      VkRenderPassBeginInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = renderPass;
      renderPassInfo.framebuffer = swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = swapChainExtent;
      /*
      we now have multiple attachments with VK_ATTACHMENT_LOAD_OP_CLEAR,
      we also need to specify multiple clear values
       */
      std::array<VkClearValue, 2> clearValues = {};
      clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
      clearValues[1].depthStencil = {1.0f, 0};
      renderPassInfo.clearValueCount =
          static_cast<uint32_t>(clearValues.size());
      renderPassInfo.pClearValues = clearValues.data();
      /*
      - command buffer to record the command to
      - The details of the render pass we've just provided
      - The render pass commands will be embedded in the primary command buffer
      itself
        and no secondary command buffers will be executed
      */
      vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphicsPipeline);
      VkBuffer vertexBuffers[] = {vertexBuffer};
      VkDeviceSize offsets[] = {0};
      // binding the vertex buffer during rendering operations
      vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
      vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0,
                           VK_INDEX_TYPE_UINT32);
      /*
      - Descriptor sets are not unique to graphics pipelines. Therefore we need
      to specify if we
      want to bind descriptor sets to the graphics or compute pipeline
      - layout that the descriptors are based on
      - index of the first descriptor set
      - number of sets to bind
      - array of sets to bind
      */
      vkCmdBindDescriptorSets(commandBuffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                              0, 1, &descriptorSets[i], 0, nullptr);
      vkCmdDrawIndexed(commandBuffers[i],
                       static_cast<uint32_t>(model.indices.size()), 1, 0, 0, 0);
      vkCmdEndRenderPass(commandBuffers[i]);

      if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  void createCommandPool() {

    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    // Command buffers are executed by submitting them on one of the device
    // queues
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // We're going to record commands for drawing, which is why we've chosen the
    // graphics queue family.
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = 0; // Optional

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  void createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());
    // iterate through the image views and create framebuffers from them
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      std::array<VkImageView, 2> attachments = {swapChainImageViews[i],
                                                depthImageView};

      VkFramebufferCreateInfo framebufferInfo = {};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      // specify with which renderPass the framebuffer needs to be compatible
      framebufferInfo.renderPass = renderPass;
      framebufferInfo.attachmentCount =
          static_cast<uint32_t>(attachments.size());
      framebufferInfo.pAttachments = attachments.data();
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                              &swapChainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  }

  /*      We need to tell Vulkan about the framebuffer attachments that will be
          used while rendering. We need to specify how many color and depth
     buffers
          there will be, how many samples to use for each of them and how their
          contents should be handled throughout the rendering operations */
  void createRenderPass() {
    VkAttachmentDescription colorAttachment = {};
    // a single color buffer attachment represented by one of the images from
    // the swap chain
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    // determine what to do with the data in the attachment before rendering and
    // after rendering.
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // Clear the values to
                                                          // a constant at the
                                                          // start
    colorAttachment.storeOp =
        VK_ATTACHMENT_STORE_OP_STORE; // Rendered contents will be stored in
                                      // memory and can be read later
    // Our application won't do anything with the stencil buffer, so the results
    // of loading and storing are irrelevant
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // The initialLayout specifies which layout the image will have before the
    // render pass begins
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // The finalLayout specifies the layout to automatically transition to when
    // the render pass finishes
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    /*
    A single render pass can consist of multiple subpasses. Subpasses are
    subsequent
    rendering operations that depend on the contents of framebuffers in previous
    passes,
    for example a sequence of post-processing effects that are applied one after
    another */
    VkAttachmentReference colorAttachmentRef = {};
    // which attachment to reference by its index in the attachment descriptions
    // array.
    colorAttachmentRef.attachment = 0;
    // which layout we would like the attachment to have during a subpass that
    // uses this reference.
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment = {};
    // The format should be the same as the depth image itself
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // we don't care about storing the depth data (storeOp), because it will not
    // be used after drawing has finished (this allow hardware optimization)
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // subpass definition
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    // reference to the color attachment:
    subpass.colorAttachmentCount = 1;
    // The index of the attachment in this array is directly referenced from the
    // fragment
    // shader with the layout(location = 0) out vec4 outColor directive!
    subpass.pColorAttachments = &colorAttachmentRef;
    // a subpass can only use a single depth (+stencil) attachment
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    /*          The subpasses in a render pass automatically take care of image
       layout transitions.
                These transitions are controlled by subpass dependencies, which
       specify memory and
                execution dependencies between subpasses. */
    VkSubpassDependency dependency = {};
    // Indices of the dependency and the dependent subpass
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    // The operations to wait on and the stages in which these operations occur.
    // We need to wait for the swap chain to finish reading from the image
    // before we can access it
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    /*          The operations that should wait on this are in the color
       attachment stage
                and involve the reading and writing of the color attachment */
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment,
                                                          depthAttachment};
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void createGraphicsPipeline() {
    auto vertShaderCode = readFile("shaders/vert.spv");
    auto fragShaderCode = readFile("shaders/frag.spv");

    /* Shader modules are just a thin wrapper around the shader bytecode that
    we've previously
    loaded from a file and the functions defined in it */
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    // To actually use the shaders we'll need to assign them to a specific
    // pipeline stage
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    // tell Vulkan in which pipeline stage the shader is going to be used.
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    // the shader module containing the code
    vertShaderStageInfo.module = vertShaderModule;
    // the function to invoke, known as the entrypoint
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

    // set up the graphics pipeline to accept vertex data
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    // structure to the format of the vertex data that will be passed to the
    // vertex shader
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions =
        &bindingDescription; // Optional
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    ;
    vertexInputInfo.pVertexAttributeDescriptions =
        attributeDescriptions.data(); // Optional

    // what kind of geometry will be drawn from the vertices and if primitive
    // restart should be enabled
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // the region of the framebuffer that the output will be rendered to.
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    // the size of the swap chain and its images may differ from the WIDTH and
    // HEIGHT of the window.
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    /* While viewports define the transformation from the image to the
    framebuffer,
    scissor rectangles define in which regions pixels will actually be stored.
    Any pixels outside the scissor rectangles will be discarded by the
    rasterizer */
    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent =
        swapChainExtent; // we want to draw to the entire framebuffer

    // viewport and scissor rectangle need to be combined into a viewport state
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    /*          The rasterizer takes the geometry that is shaped by the vertices
       from the vertex
                shader and turns it into fragments to be colored by the fragment
       shader. */
    /*          It also performs depth testing, face culling and the scissor
       test, and it can be configured
                to output fragments that fill entire polygons or just the edges
       (wireframe rendering). */
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE; // fragments that are beyond the
                                            // near and far planes are discarded
    rasterizer.rasterizerDiscardEnable =
        VK_FALSE; // the geometry is passed through the rasterizer stage
    // how fragments are generated for geometry
    rasterizer.polygonMode =
        VK_POLYGON_MODE_FILL; // fill the area of the polygon with fragments -
                              // if VK_POLYGON_MODE_LINE, we render wireframes
    rasterizer.lineWidth = 1.0f;
    // type of face culling to use - checks all the faces that are front facing
    // towards
    // the viewer and renders those while discarding all the faces
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    // the vertex order for faces to be considered front-facing and can be
    // clockwise or counterclockwise
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f;          // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional

    /*             struct configuring multisampling, which is one of the ways to
       perform anti-aliasing.
                It works by combining the fragment shader results of multiple
       polygons that rasterize
                to the same pixel. This mainly occurs along edges, which is also
       where the most noticeable
                aliasing artifacts occur. Because it doesn't need to run the
       fragment shader multiple
                times if only one polygon maps to a pixel, it is significantly
       less expensive than simply
                rendering to a higher resolution and then downscaling. */
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;          // Optional
    multisampling.pSampleMask = nullptr;            // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE;      // Optional

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    // specifies if the depth of new fragments should be compared to the depth
    // buffer to see if they should be discarded
    depthStencil.depthTestEnable = VK_TRUE;
    // specifies if the new depth of fragments that pass the depth test should
    // actually be written to the depth buffer.
    // This is useful for drawing transparent object
    depthStencil.depthWriteEnable = VK_TRUE;
    // specifies the comparison that is performed to keep or discard fragments -
    // lower depth = closer
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    // this test allows you to only keep fragments that fall within the
    // specified depth range
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    // stencil buffer operations
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {};  // Optional

    /* After a fragment shader has returned a color, it needs to be combined
    with
    the color that is already in the framebuffer. */
    /*             The most common way to use color blending is to implement
       alpha blending,
                where we want the new color to be blended with the old color
       based on its opacity */

    /* this settings are implementing alpha blending
    finalColor.rgb = newAlpha * newColor + (1 - newAlpha) * oldColor;
    finalColor.a = newAlpha.a; */
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    // set blend constants that you can use as blend factors in the color
    // blending
    // calculations specified above
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    /*             A limited amount of the state that we've specified in the
       previous
                structs can actually be changed without recreating the pipeline.
                Examples are the size of the viewport, line width and blend
       constants. To do that
                we need this structure */
    /*             VkDynamicState dynamicStates[] = {
                    VK_DYNAMIC_STATE_VIEWPORT,
                    VK_DYNAMIC_STATE_LINE_WIDTH
                };

                VkPipelineDynamicStateCreateInfo dynamicState = {};
                dynamicState.sType =
       VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
                dynamicState.dynamicStateCount = 2;
                dynamicState.pDynamicStates = dynamicStates; */

    /*          The structure also specifies push constants,
                which are another way of passing dynamic values to shaders */
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    /*
    Combining:
    -  The shader modules that define the functionality of the programmable
    stages of the graphics pipeline
    -  All of the structures that define the fixed-function stages of the
    pipeline, like input assembly, rasterizer, viewport and color blending
    -  The uniform and push values referenced by the shader that can be updated
    at draw time
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
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr; // Optional

    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1;              // Optional

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    /*          The compilation and linking of the SPIR-V bytecode to machine
       code for execution by the GPU
                doesn't happen until the graphics pipeline is created.
                So we're allowed to destroy the shader modules again as soon as
       pipeline creation is finished, */
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
  }
  // we have to wrap the shader code in a VkShaderModule
  VkShaderModule createShaderModule(const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule; // return the wrapper around the shader bytecode
  }

  void createImageViews() {
    // resize the list to fit all of the image views we'll be creating
    swapChainImageViews.resize(swapChainImages.size());
    for (uint32_t i = 0; i < swapChainImages.size(); i++) {
      swapChainImageViews[i] = createImageView(
          swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
  }

  void cleanupSwapChain() {

    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);

    for (auto framebuffer : swapChainFramebuffers) {
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    vkFreeCommandBuffers(device, commandPool,
                         static_cast<uint32_t>(commandBuffers.size()),
                         commandBuffers.data());

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto imageView : swapChainImageViews) {
      vkDestroyImageView(device, imageView, nullptr);
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

  /* Vulkan will usually just tell us that the swap chain is no longer adequate
     during presentation
     In that case we will need to call this function
  */
  void recreateSwapChain() {
    // special kind of window resizing: when framebuffer size is 0
    int width = 0, height = 0;
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    vkDeviceWaitIdle(
        device); // we shouldn't touch resources that may still be in use

    // make sure that the old versions of these objects are cleaned up before
    // recreating them
    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createDepthResources();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
  }

  void createDescriptorSets() {

    /*
    specify the descriptor pool to allocate from, the number of descriptor sets
    to allocate,
    and the descriptor layout to base them on
    */
    // we will create one descriptor set for each swap chain image, all with the
    // same layout.
    std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(),
                                               descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount =
        static_cast<uint32_t>(swapChainImages.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(swapChainImages.size());

    // You don't need to explicitly clean up descriptor sets, because they will
    // be automatically
    // freed when the descriptor pool is destroyed
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor sets!");
    }
    /* The descriptor sets have been allocated now, but the descriptors
    within still need to be configured*/

    for (size_t i = 0; i < swapChainImages.size(); i++) {
      VkDescriptorBufferInfo bufferInfo = {};
      bufferInfo.buffer = uniformBuffers[i];
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject);
      // bind the actual image and sampler resources to the descriptors in the
      // descriptor set
      VkDescriptorImageInfo imageInfo = {};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = textureImageView;
      imageInfo.sampler = textureSampler;

      std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

      VkWriteDescriptorSet descriptorWrite = {};
      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      // The first two fields specify the descriptor set to update and the
      // binding
      descriptorWrites[0].dstSet = descriptorSets[i];
      descriptorWrites[0].dstBinding = 0;
      /* We need to specify the type of descriptor again. It's possible
      to update multiple descriptors at once in an array, starting at
      index dstArrayElement. The descriptorCount field specifies how
      many array elements you want to update. */
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      // It depends on the type of descriptor which one of the three you
      // actually need to use
      descriptorWrites[0].descriptorCount = 1;
      // is used for descriptors that refer to buffer data
      descriptorWrites[0].pBufferInfo = &bufferInfo;

      descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[1].dstSet = descriptorSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].pImageInfo = &imageInfo;

      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(descriptorWrites.size()),
                             descriptorWrites.data(), 0, nullptr);
    }
  }

  void createSwapChain() {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
    // how many images we would like to have in the swap chain
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    // make sure to not exceed the maximum number of images
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
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
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                     indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0;     // Optional
      createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    // retrieve the handles
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                            swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
  }

  void createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  void createLogicalDevice() {

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    // we need to have multiple VkDeviceQueueCreateInfo structs to create a
    // queue from both families.
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};

    // priorities to queues to influence the scheduling of command buffer
    // execution
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo = {};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // the set of device features that we'll be using
    VkPhysicalDeviceFeatures deviceFeatures = {};
    // anisotropic filtering is actually an optional device feature
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
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

    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }
  // evaluate the suitability of a device
  bool isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    bool extensionsSupported = checkDeviceExtensionSupport(device);
    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }
    // to check if anisotropic filtering is available
    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return indices.isComplete() && extensionsSupported && swapChainAdequate &&
           supportedFeatures.samplerAnisotropy;
  }
  // enumerate the extensions and check if all of the required extensions are
  // amongst them.
  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;
    // determining the supported capabilitie
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                              &details.capabilities);
    // querying the supported surface formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                           details.formats.data());
    }
    // querying the supported presentation modes
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                              &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
  }

  // every operation in Vulkan, anything from drawing to uploading textures,
  // requires commands to be submitted to a queue.
  // We need to check which queue families are supported by the device and which
  // one of these supports the commands that we want to use.
  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueCount > 0 &&
          queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }
      // look for a queue family that has the capability of presenting to our
      // window surface
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
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Viewer";
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
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();

      populateDebugMessengerCreateInfo(debugCreateInfo);
      createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    } else {
      createInfo.enabledLayerCount = 0;

      createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
  }

  void populateDebugMessengerCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers)
      return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                     &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }

  // function that will return the required list of extensions based
  // on whether validation layers are enabled or not
  std::vector<const char *> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(glfwExtensions,
                                         glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
  }

  // checks if all of the requested layers are available
  bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers) {
      bool layerFound = false;
      for (const auto &layerProperties : availableLayers) {
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
  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats) {

    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }

    return availableFormats[0];
  }

  // The presentation mode is arguably the most important setting for the swap
  // chain,
  // because it represents the actual conditions for showing images to the scree
  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &availablePresentModes) {
    VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto &availablePresentMode : availablePresentModes) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return availablePresentMode;
      } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
        bestMode = availablePresentMode;
      }
    }

    return bestMode;
  }
  // The swap extent is the resolution of the swap chain images and it's
  // almost always exactly equal to the resolution of the window that we're
  // drawing to
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    } else {
      /*              Handle window resizes properly, we also need to query the
         current size of the framebuffer
                      to make sure that the swap chain images have the (new)
         right size.  */
      int width, height;
      glfwGetFramebufferSize(window, &width, &height);

      VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)};

      actualExtent.width = std::max(
          capabilities.minImageExtent.width,
          std::min(capabilities.maxImageExtent.width, actualExtent.width));
      actualExtent.height = std::max(
          capabilities.minImageExtent.height,
          std::min(capabilities.maxImageExtent.height, actualExtent.height));

      return actualExtent;
    }
  }

  // To relay the debug messages back to our program we need to setup a
  // debug messenger with a callback

  // debug callback function.  The VKAPI_ATTR and VKAPI_CALL ensure that
  // the function has the right signature for Vulkan to call it.
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
  }
};

#endif