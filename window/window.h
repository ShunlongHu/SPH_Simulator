//
// Created by QIAQIA on 2024/9/4.
//

#ifndef TUTORIALS_WINDOW_H
#define TUTORIALS_WINDOW_H
#include <functional>
namespace Sph {

class Window {
public:
    Window();
    ~Window();
    void Run(const std::function<void(void)>& func);
};

}// namespace Sph

#endif//TUTORIALS_WINDOW_H
