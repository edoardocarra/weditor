#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    // The position of each vertex is accessed from the constant array in the 
    // shader and combined with dummy z and w components to produce a position in clip coordinates
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0); //The built-in variable gl_Position functions as the output
    // we just need to pass these per-vertex colors to the fragment shader so it 
    // can output their interpolated values to the framebuffer
    fragColor = colors[gl_VertexIndex];
}