//
// Created by QIAQIA on 2024/9/4.
//

#ifndef TUTORIALS_VBO_H
#define TUTORIALS_VBO_H
#include <GL/glew.h>
namespace Sph {

class Vbo {
public:
    Vbo(GLenum target, GLenum usage);
    ~Vbo();
    void Bind() const;
    void Data(GLsizeiptr size, const void* data) const;
    GLuint id_{0};
    GLenum target_;
    GLenum usage_;
};

}// namespace Sph

#endif//TUTORIALS_VBO_H
