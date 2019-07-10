#version 450
#extension GL_ARB_separate_shader_objects : enable

const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;

/* The triangle that is formed by the positions from the vertex shader 
fills an area on the screen with fragments. The fragment shader is invoked 
on these fragments to produce a color and depth for the framebuffer  */

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 eyePos;
layout(location = 3) in vec2 resolution;
layout(location = 4) in float fov;
layout(location = 5) in vec3 normal;
layout(location = 6) in vec3 fragPos;
layout(location = 7) in vec3 light_position;
layout(location = 8) in vec3 light_color;
layout(location = 9) in float light_intensity;

//A combined image sampler descriptor is represented in GLSL by a sampler uniform
layout(binding = 1) uniform sampler2D texSampler;

// You have to specify your own output variable for each framebuffer 
layout(location = 0) out vec4 outColor;

void main() {

    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(light_position - fragPos);

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 ambient = light_intensity * light_color;
    vec3 diffuse = diff * light_color;

    //vec3 result = (ambient + diffuse) * vec3(fragColor * texture(texSampler, fragTexCoord).rgb);

    outColor = vec4(norm, 1.0);

}