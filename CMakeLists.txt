# CMake entry point
cmake_minimum_required(VERSION 3.10)
project(Tutorials)

include(cmake/find_opengl.cmake)

include_directories(window)
include_directories(gl_utils)
include_directories(external/thread_pool)
include_directories(engine)
include_directories(renderer)

add_subdirectory(window)
add_subdirectory(gl_utils)
add_subdirectory(engine)
add_subdirectory(renderer)


add_executable(main main.cpp
)
target_link_libraries(main
        window
        gl_utils
        renderer
        ${ALL_LIBS}
)