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
#include "engine.h"
#include "mesh.h"
#include "shader.hpp"
using namespace glm;

using namespace std;
namespace Sph {

static unique_ptr<MeshColorPosStrip> pParticle;
static double lastTime{0};
static GLuint programID{0};
static GLuint ViewProjMatrixID{0};
static Engine2D engine2D(1000);

void Render::Init() {
    // Initialize GLEW
    glewExperimental = true;// Needed for core profile
    if (glewInit() != GLEW_OK) {
        cerr << "Failed to initialize GLEW" << endl;
        exit(-1);
    }
    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glEnable(GL_DEPTH_TEST);

    pParticle = make_unique<MeshColorPosStrip>();
    pParticle->Data({
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
    });

    // Create and compile our GLSL program from the shaders
    programID = LoadShaders("shader/particle.vs", "shader/particle.fs");

    // Vertex shader
    ViewProjMatrixID = glGetUniformLocation(programID, "VP");
}

void Render::Step() {
    for (int i = 0; i < 10; ++i) { engine2D.Step(); }

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
    glUniformMatrix4fv(ViewProjMatrixID, 1, GL_FALSE, &ViewProjectionMatrix[0][0]);
    //    pParticle->Draw({0, 0, 0, 1, 0.5, 0.5, 0.5, 0.5}, {1, 1, 1, 1, 0, 1, 1, 1});
    pParticle->Draw(engine2D.GetXyzs(), engine2D.GetColor());
}
}// namespace Sph