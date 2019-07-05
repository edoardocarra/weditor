#ifndef RAYTRACE_H
#define RAYTRACE_H

#include <glm/glm.hpp>
#include <math.h>

struct ray {

    public:
        ray() {}
        ray(const glm::vec3& a, const glm::vec3& b) { A = a; B = b }
        glm::vec3 origin() const {return A;}
        glm::vec3 direction() const {return B;}
        glm::vec3 point_at_parameter(float t) const { return A + t*B; }
        glm::vec3 A;
        glm::vec3 B;

};

#endif