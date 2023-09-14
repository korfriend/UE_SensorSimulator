#version 430
precision highp float;
precision highp int;

in vec2 l_texcoord0;
out vec4 p3d_FragColor;

uniform sampler2D tex;

void main()
{
    vec4 colorOut = texture(tex, l_texcoord0);
    p3d_FragColor = colorOut;
}