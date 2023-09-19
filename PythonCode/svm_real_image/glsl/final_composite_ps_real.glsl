#version 430
precision highp float;
precision highp int;

in vec2 l_texcoord0;
out vec4 p3d_FragColor;

uniform sampler2D texGeoInfo2; // from sceneObj
uniform sampler2D texInterResult;

void main()
{
    // uintBitsToFloat
    const ivec2 texIdx2d = ivec2(l_texcoord0 * 1024);
    vec4 sceneColor = texelFetch(texGeoInfo2, texIdx2d, 0);

    vec4 colorOut = texture(texInterResult, l_texcoord0);

    if (sceneColor.r + sceneColor.g + sceneColor.b > 0)
        colorOut = sceneColor;

    p3d_FragColor = colorOut;
}