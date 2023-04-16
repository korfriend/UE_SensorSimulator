#version 430
precision highp float;
precision highp int;

in vec2 l_texcoord0;
out vec4 p3d_FragColor;

uniform usampler2D texGeoInfo0;
uniform usampler2D texGeoInfo1;
uniform sampler2D texGeoInfo2; // from sceneObj

uniform sampler2DArray cameraImgs;
uniform isampler2DArray semanticImgs;

void main()
{
    // uintBitsToFloat
    const ivec2 texIdx2d = ivec2(l_texcoord0 * 1024);
    uvec4 geoInfo0 = texelFetch(texGeoInfo0, texIdx2d, 0);
    uvec4 geoInfo1 = texelFetch(texGeoInfo1, texIdx2d, 0);
    vec4 sceneColor = texelFetch(texGeoInfo2, texIdx2d, 0);
    
    vec3 pos = uintBitsToFloat(geoInfo0.xyz);
    uint count = geoInfo0.w;
    uint semantic = geoInfo1.x;
    uint camId0 = geoInfo1.y;
    uint camId1 = geoInfo1.z;

    vec4 colorOut = vec4(0, 0, 0, 1);
    if (count > 0) {
        switch(semantic) {
            case 1 : colorOut = vec4(0, 1, 0, 1); break;
            case 2 : colorOut = vec4(1, 0, 0, 1); break;
            case 3 : colorOut = vec4(0, 0, 1, 1); break;
        }
    }
        
    if (sceneColor.r + sceneColor.g + sceneColor.b > 0)
        colorOut = sceneColor;

    p3d_FragColor = colorOut;
}