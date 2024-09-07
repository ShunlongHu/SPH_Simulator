project(engine)
aux_source_directory(. ENGINE_SRC)
add_library(engine ${ENGINE_SRC})
target_link_libraries(engine PUBLIC OpenCL::OpenCL)