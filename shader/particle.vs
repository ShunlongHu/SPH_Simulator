#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 squareVertices;
layout(location = 1) in vec4 xyzs; // Position of the center of the particle and size of the square
layout(location = 2) in vec4 color; // Position of the center of the particle and size of the square

// Output data ; will be interpolated for each fragment.
out vec4 particlecolor;

// Values that stay constant for the whole mesh.
uniform mat4 VP; // Model-View-Projection matrix, but without the Model (the position is in BillboardPos; the orientation depends on the camera)

void main()
{
	float particleSize = xyzs.a; // because we encoded it this way.
	vec3 particleCenter_wordspace = xyzs.xyz;
	vec3 vertexPosition_worldspace = particleCenter_wordspace + vec3(squareVertices.x * particleSize, squareVertices.y * particleSize, squareVertices.z * particleSize);
	// Output position of the vertex
	gl_Position = VP * vec4(vertexPosition_worldspace, 1.0f);
	particlecolor = color;
}

