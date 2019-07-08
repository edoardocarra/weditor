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

//A combined image sampler descriptor is represented in GLSL by a sampler uniform
layout(binding = 1) uniform sampler2D texSampler;

// You have to specify your own output variable for each framebuffer 
layout(location = 0) out vec4 outColor;

float sphereSDF(vec3 samplePoint) {
    return length(samplePoint) - 1.0;
}
float sceneSDF(vec3 samplePoint) {
    return sphereSDF(samplePoint);
}

float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + depth * marchingDirection);
        if (dist < EPSILON) {
            return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}

vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

void main() {

    // The main function is called for every fragment 
    //Textures are sampled using the built-in texture function. It takes a sampler and coordinate as arguments
    vec3 dir = rayDirection(fov,resolution,gl_FragCoord.xy);
    float dist = shortestDistanceToSurface(eyePos, dir, MIN_DIST, MAX_DIST);

    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything
        outColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    color = vec4(1.0, 0.0, 0.0, 1.0);
    //outColor = vec4(fragColor * texture(texSampler, fragTexCoord).rgb, 1.0);

}