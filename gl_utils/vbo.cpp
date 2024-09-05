//
// Created by QIAQIA on 2024/9/4.
//

#include "vbo.h"

namespace Sph {
Vbo::Vbo(GLenum target, GLenum usage) : target_{target}, usage_(usage) {
    glGenBuffers(1, &id_);
}
void Vbo::Bind() const {
    glBindBuffer(target_, id_);
}
void Vbo::Data(GLsizeiptr size, const void* data) const {
    glBufferData(target_, size, data, usage_);
}
Vbo::~Vbo() {
    glDeleteBuffers(1, &id_);
}
}// namespace Sph