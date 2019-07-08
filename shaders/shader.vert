#version 450
#extension GL_ARB_separate_shader_objects : enable

#define PI 3.1415926535898 

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec2 resolution;
} ubo;

layout(binding = 1) uniform LightObject {
    vec3 pos;
    vec3 color;
    float ambient;
} lo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 eyePos;
layout(location = 3) out vec2 resolution;
layout(location = 4) out float fov;
layout(location = 5) out vec3 normal;
layout(location = 6) out vec3 fragPos;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    eyePos = vec3(ubo.view[0].w,ubo.view[1].w,ubo.view[2].w);
    resolution = vec2(ubo.resolution.x, ubo.resolution.y);
    fov = 2.0*atan( 1.0/ubo.proj[1][1] ) * 180.0 / PI;
    fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
    normal = inNormal;
}