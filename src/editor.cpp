#include "editor.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void setupCamera(Camera& camera) {
    camera.position = glm::vec3(0,0,0);
    camera.target = glm::vec3(0,0,0);
    camera.up = glm::vec3(0,0,1);
    camera.theta = pi/2;
    camera.phi = -pi/2;
    camera.distance = 4;
}

void loadTexture(Model& model, char* filename) {
	std::cout << "[LOADING] " << filename << std::endl;

	int          texWidth = 1, texHeight = 1;
    int          texChannels = 4;
    glm::u8vec4* color       = new glm::u8vec4(155, 155, 155, 255);
    stbi_uc*     pixels      = reinterpret_cast<stbi_uc*>(color);

	Texture txt;
	if (filename!=nullptr)
		txt.pixels = stbi_load(filename, &(txt.texWidth), &(txt.texHeight), &(txt.texChannels), STBI_rgb_alpha); 
	else {
		txt.pixels = pixels;
		txt.texWidth = texWidth;
		txt.texHeight = texHeight;
	}
	model.txt = txt;
}

void loadModel(Model& model, char* filename) {
	std::cout << "[LOADING] " << filename << std::endl;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename)) {
		throw std::runtime_error(warn + err);
	}

	std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

	for (const auto& shape : shapes) {
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex = {};

			vertex.pos = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.texCoord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};

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
			face.x = idx0.vertex_index; assert(face.x >= 0);
			face.y = idx1.vertex_index; assert(face.y >= 0);
			face.z = idx2.vertex_index; assert(face.z >= 0);

			model.faces.push_back(face);
		}
	}
}

int main(int argc, char *argv[])
{

	Viewer app;

	if (argc == 1) return EXIT_FAILURE;
	
	try{
		loadModel(app.model, argv[1]);
		if(argc > 2) 
			loadTexture(app.model, argv[2]);
		else
			loadTexture(app.model, nullptr);
		setupCamera(app.camera);
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}