#version 430

// Vertex inputs
in vec3 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

// Output to fragment shader
out vec2 texcoord;
out vec3 worldcoord;

// Uniform inputs
uniform mat4 p3d_ModelViewProjectionMatrix;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vec4(p3d_Vertex, 1);
    texcoord = p3d_MultiTexCoord0;
    worldcoord = p3d_Vertex.xyz;
}