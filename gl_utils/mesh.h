//
// Created by QIAQIA on 2024/9/5.
//

#ifndef TUTORIALS_MESH_H
#define TUTORIALS_MESH_H
#include <vector>

#include "vao.h"
#include "vbo.h"
namespace Sph {

class MeshStrip {
public:
    MeshStrip();
    ~MeshStrip() = default;
    void Data(const std::vector<float>& vbVec);
    virtual void Draw();

    Vao vao_;
    Vbo vbo_;
    std::vector<float> vbVec_;
};

class MeshColorPosStrip : public MeshStrip {
public:
    MeshColorPosStrip();
    ~MeshColorPosStrip() = default;
    virtual void Draw(const std::vector<float>& xyzsVec, const std::vector<float>& colorVec);
    Vbo xo_;
    Vbo co_;
    int maxParticleNum_{65536};
};

}// namespace Sph

#endif//TUTORIALS_MESH_H
