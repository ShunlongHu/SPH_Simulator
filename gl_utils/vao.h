//
// Created by QIAQIA on 2024/9/4.
//

#ifndef TUTORIALS_VAO_H
#define TUTORIALS_VAO_H
#include <GL/glew.h>
namespace Sph {

class Vao {
public:
    Vao();
    ~Vao();
    void Bind() const;
    GLuint id_{0};
};

}// namespace Sph

#endif//TUTORIALS_VAO_H
