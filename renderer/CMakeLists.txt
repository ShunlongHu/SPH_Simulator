project(renderer)
aux_source_directory(. RENDERER_SRC)
add_library(renderer ${RENDERER_SRC})
target_link_libraries(renderer gl_utils engine)