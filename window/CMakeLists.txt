project(window)
aux_source_directory(. WINDOW_SRC)
add_library(window ${WINDOW_SRC})
target_link_libraries(window glfw)