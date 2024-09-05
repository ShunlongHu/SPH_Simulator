//
// Created by QIAQIA on 2024/9/5.
//

#include "mesh.h"

namespace Sph {
MeshStrip::MeshStrip() : vbo_(GL_ARRAY_BUFFER, GL_STATIC_DRAW) {
}
void MeshStrip::Data() {
    vao_.Bind();
    vbo_.Bind();
    vbo_.Data(sizeof(float) * vbVec_.size(), vbVec_.data());

    glEnableVertexAttribArray(0);
    vbo_.Bind();
    glVertexAttribPointer(
            0,       // attribute. No particular reason for 0, but must match the layout in the shader.
            3,       // size
            GL_FLOAT,// type
            GL_FALSE,// normalized?
            0,       // stride
            (void*) 0// array buffer offset
    );
}
}// namespace Sph