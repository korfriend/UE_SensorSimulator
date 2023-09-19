#version 430
precision highp float;
precision highp int;

in vec2 l_texcoord0;
out vec4 p3d_FragColor;

void main()
{
    vec4 colorOut = texture(tex, l_texcoord0);

    if (sceneColor.r + sceneColor.g + sceneColor.b > 0)
        colorOut = sceneColor;

    p3d_FragColor = colorOut;
}