#version 430
precision highp float;
precision highp int;

const float PI = 3.14159265359;

in vec2 l_texcoord0;
out vec4 p3d_FragColor;

uniform mat4 matViewProj0;
uniform mat4 matViewProj1;
uniform mat4 matViewProj2;
uniform mat4 matViewProj3;

uniform isampler2D texGeoInfo0;
uniform isampler2D texGeoInfo1;
uniform sampler2D texGeoInfo2; // from sceneObj

// uniform sampler2DArray cameraImgs;
uniform isampler2DArray semanticImgs;

uniform sampler2D tex;

uniform int img_w;
uniform int img_h;

uniform int debug_mode;

void main()
{
    mat4 viewProjs[4] = {matViewProj0, matViewProj1, matViewProj2, matViewProj3};

    // uintBitsToFloat
    const ivec2 texIdx2d = ivec2(l_texcoord0 * 1024);
    ivec4 geoInfo0 = texelFetch(texGeoInfo0, texIdx2d, 0);
    ivec4 geoInfo1 = texelFetch(texGeoInfo1, texIdx2d, 0);
    vec4 sceneColor = texelFetch(texGeoInfo2, texIdx2d, 0);
    
    vec3 pos = uintBitsToFloat(geoInfo1.xyz);
    int count = geoInfo0.w;
    int semantic = geoInfo0.x;
    int camId0 = geoInfo0.y;
    int camId1 = geoInfo0.z;

    vec4 colorOut = texture(tex, l_texcoord0);
    const int debugMode = debug_mode;
    if (debugMode == 0) {
        // height correction
        switch (semantic / 100) {
            case 1: pos.z = 60; break;
            case 2: pos.z = 60; break;
        }

        int overlapIndex[4] = {int(camId0), int(camId1), 0, 0};
        semantic = 0;
        for (int i = 0; i < count; i++) {
            int camId = overlapIndex[i];
            vec4 imagePos = viewProjs[camId] * vec4(pos, 1.0);
            vec2 imagePos2D = imagePos.xy / imagePos.w;
            vec2 texPos = (imagePos2D + vec2(1.0, 1.0)) * 0.5;

            ivec2 texIdx2d = ivec2((1 - texPos.x) * img_w + 0.5, (1 - texPos.y) * img_h + 0.5);
            semantic = max(semantic, texelFetch(semanticImgs, ivec3(texIdx2d, camId), 0).r);
        }

        switch (semantic) {
            case 1: colorOut += vec4(0, 0, 0.2, 1); break;
            case 2: colorOut += vec4(0.2, 0, 0, 1); break;
        }
    }

    if (sceneColor.r + sceneColor.g + sceneColor.b > 0)
        colorOut = sceneColor;

    p3d_FragColor = colorOut;
}