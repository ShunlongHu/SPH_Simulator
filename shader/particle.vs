#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 squareVertices;

// Output data ; will be interpolated for each fragment.
out vec4 particlecolor;

// Values that stay constant for the whole mesh.
uniform vec3 CameraRight_worldspace;
uniform vec3 CameraUp_worldspace;
uniform mat4 VP; // Model-View-Projection matrix, but without the Model (the position is in BillboardPos; the orientation depends on the camera)

void main()
{
	float particleSize = 1; // because we encoded it this way.
	vec3 particleCenter_wordspace = vec3(0,0,0);
	
	vec3 vertexPosition_worldspace = 
		particleCenter_wordspace
		+ CameraRight_worldspace * squareVertices.x * particleSize
		+ CameraUp_worldspace * squareVertices.y * particleSize;

	// Output position of the vertex
	// gl_Position = VP * vec4(vertexPosition_worldspace, 1.0f);
	gl_Position = vec4(squareVertices.x, squareVertices.y, 5, 1.0f);
	particlecolor = vec4(1.0f,1.0f,1.0f,0.5f);
}

