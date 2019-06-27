#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;


void main() {
    // The position of each vertex is accessed from the constant array in the 
    // shader and combined with dummy z and w components to produce a position in clip coordinates
    gl_Position = vec4(inPosition, 0.0, 1.0); //The built-in variable gl_Position functions as the output
    // we just need to pass these per-vertex colors to the fragment shader so it 
    // can output their interpolated values to the framebuffer
    fragColor = inColor;
}