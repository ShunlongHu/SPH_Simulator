//
// Created by QIAQIA on 2024/9/4.
//

#include "vao.h"

namespace Sph {
Vao::Vao() {
    glGenVertexArrays(1, &id_);
}
void Vao::Bind() const {
    glBindVertexArray(id_);
}
Vao::~Vao() {
    glDeleteVertexArrays(1, &id_);
}
}// namespace Sph