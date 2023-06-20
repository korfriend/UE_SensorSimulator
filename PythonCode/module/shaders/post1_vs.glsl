#version 430

// Vertex inputs
in vec2 p3d_MultiTexCoord0;
in vec3 p3d_Vertex;

// Output to fragment shader
out vec2 l_texcoord0;

uniform mat4 mat_modelproj;


void main()
{
    gl_Position = mat_modelproj * vec4(p3d_Vertex, 1);
    l_texcoord0 = p3d_MultiTexCoord0;
}