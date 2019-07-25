#include "editor.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void setupLight(Light &camera) {
  camera.position = glm::vec3(0.0, 5.0, 0.0);
  camera.color = glm::vec3(0.6, 0.6, 0.6);
  camera.intensity = 0.5;
}

void setupCamera(Camera &camera, const Model &model) {
  auto from = ym::vec3f{0, 2, 3};
  auto to = ym::vec3f{(model.bbox_max.x + model.bbox_min.x) / 2,
                      (model.bbox_max.y + model.bbox_min.y) / 2,
                      (model.bbox_max.z + model.bbox_min.z) / 2};
  auto up = ym::vec3f{0, 1, 0};
  camera.orthographic = false;
  camera.film.x = 0.036f;
  camera.film.y = 0.024f;
  camera.lens = 0.050;
  camera.frame = ym::lookat_frame3(from, to, up);
  camera.focus = length(from - to);
  camera.aperture = 0.01f;
}

void loadTexture(Model &model, char *filename) {
  std::cout << "[LOADING] " << filename << std::endl;

  int texWidth = 1, texHeight = 1;
  int texChannels = 4;
  glm::u8vec4 *color = new glm::u8vec4(155, 155, 155, 255);
  stbi_uc *pixels = reinterpret_cast<stbi_uc *>(color);

  Texture txt;
  if (filename != nullptr)
    txt.pixels = stbi_load(filename, &(txt.texWidth), &(txt.texHeight),
                           &(txt.texChannels), STBI_rgb_alpha);
  else {
    txt.pixels = pixels;
    txt.texWidth = texWidth;
    txt.texHeight = texHeight;
  }
  model.txt = txt;
}

void loadModel(Model &model, char *filename) {
  std::cout << "[LOADING] " << filename << std::endl;

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename)) {
    throw std::runtime_error(warn + err);
  }

  std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

  for (const auto &shape : shapes) {
    glm::vec3 bbox_min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    glm::vec3 bbox_max = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);
    for (const auto &index : shape.mesh.indices) {
      Vertex vertex = {};
      vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]};

      if (vertex.pos.x < bbox_min.x)
        bbox_min.x = vertex.pos.x;
      if (vertex.pos.y < bbox_min.y)
        bbox_min.y = vertex.pos.y;
      if (vertex.pos.z < bbox_min.z)
        bbox_min.z = vertex.pos.z;
      if (vertex.pos.x > bbox_max.x)
        bbox_max.x = vertex.pos.x;
      if (vertex.pos.y > bbox_max.y)
        bbox_max.y = vertex.pos.y;
      if (vertex.pos.z > bbox_max.z)
        bbox_max.z = vertex.pos.z;

      if (attrib.normals.size() > 0)
        vertex.normal = {attrib.normals[3 * index.normal_index + 0],
                         attrib.normals[3 * index.normal_index + 1],
                         attrib.normals[3 * index.normal_index + 2]};
      else
        vertex.normal = {0.0f, 0.0f, 0.0f};

      vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                         1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};

      vertex.color = {1.0f, 1.0f, 1.0f};
      if (uniqueVertices.count(vertex) == 0) {
        uniqueVertices[vertex] = static_cast<uint32_t>(model.vertices.size());
        model.vertices.push_back(vertex);
      }

      model.indices.push_back(uniqueVertices[vertex]);
    }

    for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
      // Get the three indexes of the face (all faces are triangular)
      tinyobj::index_t idx0 = shape.mesh.indices[3 * f + 0];
      tinyobj::index_t idx1 = shape.mesh.indices[3 * f + 1];
      tinyobj::index_t idx2 = shape.mesh.indices[3 * f + 2];

      glm::vec3 face;
      face.x = idx0.vertex_index;
      assert(face.x >= 0);
      face.y = idx1.vertex_index;
      assert(face.y >= 0);
      face.z = idx2.vertex_index;
      assert(face.z >= 0);

      model.faces.push_back(face);
    }

    // calculate normals if not present
    if (attrib.normals.size() == 0) {
      for (Vertex &vertex : model.vertices) {
        vertex.normal = glm::vec3(0.0f);
      }
      for (glm::vec3 &face : model.faces) {
        glm::vec3 vA = model.vertices[face.x].pos;
        glm::vec3 vB = model.vertices[face.y].pos;
        glm::vec3 vC = model.vertices[face.z].pos;
        glm::vec3 normal = glm::cross(vB - vA, vC - vA);
        float area = glm::length(glm::cross(vB - vA, vC - vA)) / 2;
        model.vertices[face.x].normal += normal * area;
        model.vertices[face.y].normal += normal * area;
        model.vertices[face.z].normal += normal * area;
      }
      for (Vertex &vertex : model.vertices) {
        vertex.normal = glm::normalize(vertex.normal);
      }
    }

    model.bbox_min = bbox_min;
    model.bbox_max = bbox_max;
  }
}

int main(int argc, char *argv[]) {

  Viewer app;

  if (argc == 1)
    return EXIT_FAILURE;

  try {
    loadModel(app.model, argv[1]);
    if (argc > 2)
      loadTexture(app.model, argv[2]);
    else
      loadTexture(app.model, nullptr);
    setupCamera(app.camera, app.model);
    setupLight(app.light);
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}