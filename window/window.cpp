//
// Created by QIAQIA on 2024/9/4.
//

#include "window.h"
#include <GLFW/glfw3.h>
#include <iostream>
using namespace std;
GLFWwindow* pWindow;
namespace Sph {
Window::Window() {
    // Initialize GLFW
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << endl;
        exit(-1);
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);// To make macOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a pWindow and create its OpenGL context
    pWindow = glfwCreateWindow(1024, 768, "Tutorial 18 - Particles", NULL, NULL);
    if (pWindow == nullptr) {
        cerr << "Failed to open GLFW pWindow. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials." << endl;
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(pWindow);

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(pWindow, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited movement
    glfwSetInputMode(pWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(pWindow, 1024 / 2, 768 / 2);
}

Window::~Window() {
    glfwTerminate();
}
void Window::Run(const std::function<void(void)>& func) {
    do {
        func();
        // Swap buffers
        glfwSwapBuffers(pWindow);
        glfwPollEvents();
    }// Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(pWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
           glfwWindowShouldClose(pWindow) == 0);
}
}// namespace Sph