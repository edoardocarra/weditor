#version 450
#extension GL_ARB_separate_shader_objects : enable

/* The triangle that is formed by the positions from the vertex shader 
fills an area on the screen with fragments. The fragment shader is invoked 
on these fragments to produce a color and depth for the framebuffer  */

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
//A combined image sampler descriptor is represented in GLSL by a sampler uniform
layout(binding = 1) uniform sampler2D texSampler;

// You have to specify your own output variable for each framebuffer 
layout(location = 0) out vec4 outColor;

void main() {
    // The main function is called for every fragment 
    //Textures are sampled using the built-in texture function. It takes a sampler and coordinate as arguments
    outColor = vec4(fragColor * texture(texSampler, fragTexCoord).rgb, 1.0);
}