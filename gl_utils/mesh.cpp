//
// Created by QIAQIA on 2024/9/5.
//

#include "mesh.h"

namespace Sph {
MeshStrip::MeshStrip() : vbo_(GL_ARRAY_BUFFER, GL_STATIC_DRAW) {
}
void MeshStrip::Data(const std::vector<float>& vbVec) {
    vbVec_ = vbVec;
    vao_.Bind();
    vbo_.Bind();
    vbo_.Data(sizeof(float) * vbVec_.size(), vbVec_.data());
}
void MeshStrip::Draw() {
    vao_.Bind();
    // define attrib array
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
    glDrawArrays(GL_TRIANGLE_STRIP, 0, vbVec_.size() / 3);
    glDisableVertexAttribArray(0);
}
MeshColorPosStrip::MeshColorPosStrip() : xo_(GL_ARRAY_BUFFER, GL_STREAM_DRAW), co_(GL_ARRAY_BUFFER, GL_STREAM_DRAW) {
}
void MeshColorPosStrip::Draw(const std::vector<float>& xyzsVec, const std::vector<float>& colorVec) {
    vao_.Bind();

    auto particleNum = xyzsVec.size() / 4;
    xo_.Bind();
    xo_.Data(maxParticleNum_ * sizeof(float) * 4, nullptr);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particleNum * sizeof(float) * 4, xyzsVec.data());

    co_.Bind();
    co_.Data(maxParticleNum_ * sizeof(float) * 4, nullptr);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particleNum * sizeof(float) * 4, colorVec.data());


    // define attrib array
    // 1st, mesh array
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
    // 2nd, xyzs array
    glEnableVertexAttribArray(1);
    xo_.Bind();
    glVertexAttribPointer(
            1,       // attribute. No particular reason for 0, but must match the layout in the shader.
            4,       // size
            GL_FLOAT,// type
            GL_FALSE,// normalized?
            0,       // stride
            (void*) 0// array buffer offset
    );
    // 3rd, color array
    glEnableVertexAttribArray(2);
    co_.Bind();
    glVertexAttribPointer(
            2,       // attribute. No particular reason for 1, but must match the layout in the shader.
            4,       // size : r + g + b + a => 4
            GL_FLOAT,// type
            GL_TRUE, // normalized?    *** YES, this means that the unsigned char[4] will be accessible with a vec4 (floats) in the shader ***
            0,       // stride
            (void*) 0// array buffer offset
    );

    glVertexAttribDivisor(0, 0);// particles vertices : always reuse the same 4 vertices -> 0
    glVertexAttribDivisor(1, 1);// positions : one per quad (its center)                 -> 1
    glVertexAttribDivisor(2, 1);// color : one per quad                                  -> 1

    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, vbVec_.size() / 3, particleNum);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
}
}// namespace Sph