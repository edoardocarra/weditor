#version 450
#extension GL_ARB_separate_shader_objects : enable

/* The triangle that is formed by the positions from the vertex shader 
fills an area on the screen with fragments. The fragment shader is invoked 
on these fragments to produce a color and depth for the framebuffer  */

layout(location = 0) in vec3 fragColor;

// You have to specify your own output variable for each framebuffer 
layout(location = 0) out vec4 outColor;

void main() {
    // The main function is called for every fragment 
    outColor = vec4(fragColor, 1.0);
}