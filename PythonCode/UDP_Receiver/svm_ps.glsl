#version 430

// fragment-pixel output color
out vec4 p3d_FragColor;

// input from vertex shader
in vec2 texcoord;
in vec3 worldcoord;

uniform sampler2D p3d_Texture0;
uniform sampler2D myTexture0;
uniform sampler2D myTexture1;
uniform sampler2D myTexture2;
uniform sampler2D myTexture3;

void main() {

    vec4 color = texture(myTexture0, texcoord);
    p3d_FragColor = color.rgba;
    //p3d_FragColor = texture(p3d_Texture0, texcoord);
    //p3d_FragColor = vec4(texcoord.x, texcoord.y, 0.0, 1.0);
    //p3d_FragColor = vec4(1, 0, 0.0, 1.0);
}