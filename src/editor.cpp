
#include "editor.h"

int view() {

	GLFWwindow* window;

    glm::mat4 matrix;
    glm::vec4 vec;
    auto test = matrix * vec;

	/* Initialize the library */
	if (!glfwInit())
		return -1;

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(640, 480, "CRISTOCANARO", NULL, NULL);
	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		/* Render here */
		//glClear(GL_COLOR_BUFFER_BIT);

		///* Swap front and back buffers */
		//glfwSwapBuffers(window);

		///* Poll for and process events */
		//glfwPollEvents();
	}

	glfwTerminate();

	return 0;

}



int main(void)
{
	return view();
}