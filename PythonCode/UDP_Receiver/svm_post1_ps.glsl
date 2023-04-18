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

uniform usampler2D texGeoInfo0;
uniform usampler2D texGeoInfo1;
uniform sampler2D texGeoInfo2; // from sceneObj

uniform sampler2DArray cameraImgs;
uniform isampler2DArray semanticImgs;


vec4 blendArea(int camId0, int camId1, vec3 pos, mat4 viewProjs[4], int caseId, float weightId0, int debugMode) {
    const float bias0 = weightId0;//1.0;
    const float bias1 = 1.0 - weightId0;//1.0;

    vec4 imagePos = viewProjs[camId0] * vec4(pos, 1.0);
    vec2 imagePos2D = imagePos.xy / imagePos.w;
    vec2 texPos0 = (imagePos2D + vec2(1.0, 1.0)) * 0.5;

    imagePos = viewProjs[camId1] * vec4(pos, 1.0);
    imagePos2D = imagePos.xy / imagePos.w;
    vec2 texPos1 = (imagePos2D + vec2(1.0, 1.0)) * 0.5;

    float u0 = texPos0.x;
    float u1 = 1.0 - texPos1.x;
    float v0 = texPos0.y;
    float v1 = texPos1.y;

    float w0 = u0, w1 = u1;
    if (u0 > v0 && u1 > v1) 
    {
        w0 = v0;// + u0;
        w1 = v1;// + u1;
    }
    w0 *= bias0;
    w1 *= bias1;

    float unitW0 = w0 / (w0 + w1);
    float unitW1 = w1 / (w0 + w1);

    w0 = sin(unitW0 * PI * 0.5);
    w0 *= w0;
    w1 = 1.0 - w0; // based on the sin eq of sin^2 + cos^2 = 1
    //w1 = cos(unitW0 * PI * 0.5);
    //w1 *= w1;

    // the same as the above
    //w0 = (sin((unitW0 - 0.5) * PI) + 1) * 0.5;
    //w1 = 1.0 - w0;

    vec4 colorOut;
    if(debugMode == 1) {
        switch (caseId) {
            case 1: colorOut = vec4(w0, w1, 0, 1); break;
            case 4: colorOut = vec4(0, w0, w1, 1); break;
            case 7: colorOut = w0 * vec4(0, 0, 1, 1) + w1 * vec4(1, 1, 1, 1); break;
            case 3: colorOut = w0 * vec4(1, 1, 1, 1) + w1 * vec4(1, 0, 0, 1); break;
        }
    }
    else {
        vec4 img1 = texture(cameraImgs, vec3(1 - texPos0.x, 1 - texPos0.y, camId0));
        vec4 img2 = texture(cameraImgs, vec3(1 - texPos1.x, 1 - texPos1.y, camId1));
        colorOut = w0 * img1 + w1 * img2;

        ivec2 texIdx2d0 = ivec2((1 - texPos0.x) * 500 + 0.5, (1 - texPos0.y) * 300 + 0.5);
        int semantic0 = texelFetch(semanticImgs, ivec3(texIdx2d0, camId0), 0).r;
        ivec2 texIdx2d1 = ivec2((1 - texPos1.x) * 500 + 0.5, (1 - texPos1.y) * 300 + 0.5);
        int semantic1 = texelFetch(semanticImgs, ivec3(texIdx2d1, camId1), 0).r;
        if(pos.z == 100.0) {
            if(semantic0 != 3 && semantic1 != 3) {
                colorOut = vec4(0, 0, 0, 1);
            }
        }

        //colorOut.rgb *= 255.0;
        //colorOut.r = (int(colorOut.r + 2.9)) / 255.0;
        //colorOut.g = (int(colorOut.g + 2.9)) / 255.0;
        //colorOut.b = (int(colorOut.b + 2.9)) / 255.0;
    }

    colorOut.a = 1;
    return colorOut; 
}

void main()
{
    mat4 viewProjs[4] = {matViewProj0, matViewProj1, matViewProj2, matViewProj3};

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
#define MYDEBUG__
#ifdef MYDEBUG__ 
    /**/
    if (count > 0) {
        switch(semantic) {
            case 1 : colorOut = vec4(0, 1, 0, 1); break;
            case 2 : colorOut = vec4(1, 0, 0, 1); break;
            case 3 : colorOut = vec4(0, 0, 1, 1); break;
            default : colorOut = vec4(1, 1, 1, 1); break;
        }
    }
    /**/
    switch(count) {
        case 1 : colorOut = vec4(0, 1, 0, 1); break;
        case 2 : colorOut = vec4(1, 0, 0, 1); break;
        default : colorOut = vec4(1, 1, 1, 1); break;
    }/**/
#else
    
    // height correction
    switch (semantic) {
        case 2: pos.z = 25; break;
        case 3: pos.z = 0; break;
    }

    int overlapIndex[4] = {int(camId0), int(camId1), 0, 0};
    const int debugMode = 0;
    switch (count) {
        case 1: {
            if(debugMode == 1) {
                switch(overlapIndex[0]) {
                    case 0: colorOut = vec4(1, 0, 0, 1); break;
                    case 1: colorOut = vec4(0, 1, 0, 1); break;
                    case 2: colorOut = vec4(0, 0, 1, 1); break;
                    case 3: colorOut = vec4(1, 1, 1, 1); break;
                }
            }
            else {
                int camId = overlapIndex[0];
                vec4 imagePos = viewProjs[camId] * vec4(pos, 1.0);
                vec2 imagePos2D = imagePos.xy / imagePos.w;
                vec2 texPos0 = (imagePos2D + vec2(1.0, 1.0)) * 0.5;
                colorOut = texture(cameraImgs, vec3(1 - texPos0.x, 1 - texPos0.y, camId));
            }
            break;
        }
        case 2: {
            int idx0 = overlapIndex[0];
            int idx1 = overlapIndex[1];
            int enc = idx0 * 2 + idx1;

            // TO DO : determine dynamic blending weights
            float w01 = 0.5;
            float w12 = 0.5;
            float w23 = 0.9;
            float w30 = 0.1;
            switch(enc) {
                case 1: {
                    colorOut = blendArea(0, 1, pos, viewProjs, enc, w01, debugMode);
                    break;
                }
                case 4: {
                    colorOut = blendArea(1, 2, pos, viewProjs, enc, w12, debugMode);
                    break;
                }
                case 7: {
                    colorOut = blendArea(2, 3, pos, viewProjs, enc, w23, debugMode);
                    break;
                }
                case 3: {
                    colorOut = blendArea(3, 0, pos, viewProjs, enc, w30, debugMode);
                    break;
                }
            }
            break;
        }
    }
#endif
    //if (sceneColor.r + sceneColor.g + sceneColor.b > 0)
    //    colorOut = sceneColor;

    p3d_FragColor = colorOut;
}