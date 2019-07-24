#version 450
#extension GL_ARB_separate_shader_objects : enable

#define PI 3.1415926535898

layout(binding = 0) uniform UniformBufferObject {
  // camera
  mat4 model;
  mat4 view;
  mat4 proj;
  vec2 resolution;
  // light
  vec3 light_position;
  vec3 light_color;
  float light_intensity;
  vec3 V1;
  vec3 V2;
  vec3 V3;
}
ubo;

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
layout(location = 7) out vec3 light_position;
layout(location = 8) out vec3 light_color;
layout(location = 9) out float light_intensity;

void main() {
  gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

  vec4 V1_Position = ubo.proj * ubo.view * ubo.model * vec4(ubo.V1, 1.0);
  vec4 V2_Position = ubo.proj * ubo.view * ubo.model * vec4(ubo.V2, 1.0);
  vec4 V3_Position = ubo.proj * ubo.view * ubo.model * vec4(ubo.V3, 1.0);

  fragColor = (V1_Position == gl_Position || V2_Position == gl_Position || V3_Position == gl_Position) ? vec3(1,0,0) : inColor;
  //fragColor = vec3(sqrt((inPosition.x - ubo.V1.x)*(inPosition.x - ubo.V1.x) + (inPosition.y - ubo.V1.y)*(inPosition.y - ubo.V1.y) + (inPosition.z - ubo.V1.z)*(inPosition.z - ubo.V1.z)),0,0);
  fragTexCoord = inTexCoord;
  eyePos = vec3(ubo.view[0].w, ubo.view[1].w, ubo.view[2].w);
  resolution = vec2(ubo.resolution.x, ubo.resolution.y);
  fov = 2.0 * atan(1.0 / ubo.proj[1][1]) * 180.0 / PI;
  fragPos = vec3(ubo.model * vec4(inPosition, 1.0));
  normal = inNormal;
  light_position = ubo.light_position;
  light_color = ubo.light_color;
  light_intensity = ubo.light_intensity;
}