//
// Created by QIAQIA on 2024/9/5.
//

#ifndef TUTORIALS_MESH_H
#define TUTORIALS_MESH_H
#include "vao.h"
#include "vbo.h"
#include <vector>
namespace Sph {

class MeshStrip {
public:
    MeshStrip();
    ~MeshStrip() = default;
    void Data();

    Vao vao_;
    Vbo vbo_;

    std::vector<float> vbVec_;
};

}// namespace Sph

#endif//TUTORIALS_MESH_H
