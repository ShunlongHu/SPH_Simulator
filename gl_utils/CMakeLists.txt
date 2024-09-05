project(gl_utils)
aux_source_directory(. GL_UTILS_SRC)
add_library(gl_utils ${GL_UTILS_SRC})
target_link_libraries(gl_utils PUBLIC ${OPENGL_LIBRARY}
        GLEW_1130)