//
// Created by QIAQIA on 2024/9/5.
//

#include "render.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <iostream>
#include <memory>

#include "controls.hpp"
#include "mesh.h"
#include "shader.hpp"
using namespace glm;

using namespace std;
namespace Sph {

static unique_ptr<MeshStrip> pParticle;
static double lastTime{0};
static GLuint programID{0};
static GLuint CameraRight_worldspace_ID{0};
static GLuint CameraUp_worldspace_ID{0};
static GLuint ViewProjMatrixID{0};

void Render::Init() {
    // Initialize GLEW
    glewExperimental = true;// Needed for core profile
    if (glewInit() != GLEW_OK) {
        cerr << "Failed to initialize GLEW" << endl;
        exit(-1);
    }
    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it is closer to the camera than the former one
    glDepthFunc(GL_LESS);

    pParticle = make_unique<MeshStrip>();
    pParticle->vbVec_ = {
            -0.5f,
            -0.5f,
            0.0f,
            0.5f,
            -0.5f,
            0.0f,
            -0.5f,
            0.5f,
            0.0f,
            0.5f,
            0.5f,
            0.0f,
    };

    // Create and compile our GLSL program from the shaders
    programID = LoadShaders("shader/particle.vs", "shader/particle.fs");

    // Vertex shader
    CameraRight_worldspace_ID = glGetUniformLocation(programID, "CameraRight_worldspace");
    CameraUp_worldspace_ID = glGetUniformLocation(programID, "CameraUp_worldspace");
    ViewProjMatrixID = glGetUniformLocation(programID, "VP");
}

void Render::Step() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    double currentTime = glfwGetTime();
    double delta = currentTime - lastTime;
    lastTime = currentTime;

    computeMatricesFromInputs();
    glm::mat4 ProjectionMatrix = getProjectionMatrix();
    glm::mat4 ViewMatrix = getViewMatrix();
    glm::mat4 ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Use our shader
    glUseProgram(programID);
    glUniform3f(CameraRight_worldspace_ID, ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
    glUniform3f(CameraUp_worldspace_ID, ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);
    glUniformMatrix4fv(ViewProjMatrixID, 1, GL_FALSE, &ViewProjectionMatrix[0][0]);

    // define attrib array
    glEnableVertexAttribArray(0);
    pParticle->vbo_.Bind();
    glVertexAttribPointer(
            0,       // attribute. No particular reason for 0, but must match the layout in the shader.
            3,       // size
            GL_FLOAT,// type
            GL_FALSE,// normalized?
            0,       // stride
            (void*) 0// array buffer offset
    );

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray(0);
}
}// namespace Sph