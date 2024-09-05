#include "render.h"
#include "window.h"
#include <iostream>

using namespace Sph;
int main() {
    Window window;
    Render::Init();
    window.Run([]() { Render::Step(); });
    return 0;
}
