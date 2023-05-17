#version 430
//#extension GL_EXT_texture_array
//#extension GL_NV_texture_array
const float PI = 3.14159265359;
// fragment-pixel output color
out vec4 p3d_FragColor;

// input from vertex shader
in vec2 texcoord;
in vec3 worldcoord;

//uniform mat4 matViewProjs[4];
uniform mat4 matViewProj0;
uniform mat4 matViewProj1;
uniform mat4 matViewProj2;
uniform mat4 matViewProj3;

uniform vec4 camPositions[4];

uniform sampler2DArray cameraImgs;
uniform sampler2DArray semanticImgs;

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
        //colorOut.rgb *= 255.0;
        //colorOut.r = (int(colorOut.r + 2.9)) / 255.0;
        //colorOut.g = (int(colorOut.g + 2.9)) / 255.0;
        //colorOut.b = (int(colorOut.b + 2.9)) / 255.0;
    }

    colorOut.a = 1;
    return colorOut; 
}

void main() {
    vec3 pos = worldcoord;
    //pos.z = 100;
    mat4 viewProjs[4] = {matViewProj0, matViewProj1, matViewProj2, matViewProj3};

    vec4 colorOut = vec4(0); 
    
    int count = 0;
    int overlapIndex[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        //vec3 postemp= pos;
        //if( i == 3 || i == 0) postemp.z = 100;
        vec4 imagePos = viewProjs[i] * vec4(pos, 1.0);
        vec3 imagePos3 = imagePos.xyz / imagePos.w;
        vec2 texPos = (imagePos3.xy + vec2(1.0, 1.0)) * 0.5;
        if (imagePos3.z >= 0.0 && imagePos3.z <= 1.0
            && texPos.x >= 0.0 && texPos.x <= 1.0
            && texPos.y >= 0.0 && texPos.y <= 1.0) {
                overlapIndex[count++] = i;
        }
    }

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
                int semantic = int(texture(semanticImgs, vec3(1 - texPos0.x, 1 - texPos0.y, camId)).r * 255.0 + 0.5);
                if(semantic > 0) colorOut = vec4(1, 0, 0, 1);
            }
            break;
        }
        case 2: {
            int idx0 = overlapIndex[0];
            int idx1 = overlapIndex[1];
            //vec3 camPos0 = camPositions[idx0].xyz;
            //vec3 camPos1 = camPositions[idx1].xyz;
            //vec2 vecX0 = normalize(pos.xz - camPos0.xz);
            //vec2 vecX1 = normalize(pos.xz - camPos1.xz);
            //vec2 vecY0 = normalize(pos.yz - camPos1.yz);
            //vec2 vecY1 = normalize(pos.yz - camPos1.yz);
            //float angleX = acos(dot(vecX0, vecY0));
            //float angleY = acos(dot(vecX1, vecY1));

            int enc = idx0 * 2 + idx1;
            switch(enc) {
                case 1: {
                    colorOut = blendArea(0, 1, pos, viewProjs, enc, 0.9, debugMode);
                    break;
                }
                case 4: {
                    colorOut = blendArea(1, 2, pos, viewProjs, enc, 0.5, debugMode);
                    break;
                }
                case 7: {
                    colorOut = blendArea(2, 3, pos, viewProjs, enc, 0.5, debugMode);
                    break;
                }
                case 3: {
                    colorOut = blendArea(3, 0, pos, viewProjs, enc, 0.1, debugMode);
                    break;
                }
            }
            break;
        }
    }
    p3d_FragColor = colorOut;
}

void main2() {
    vec3 pos = worldcoord;
    mat4 viewProjs[4] = {matViewProj0, matViewProj1, matViewProj2, matViewProj3};

    vec4 colorOut = vec4(0); 
    
    for (int i = 0; i < 4; i++) {
        vec4 imagePos = viewProjs[i] * vec4(pos, 1.0);
        vec3 imagePos3 = imagePos.xyz / imagePos.w;
        vec2 texPos = (imagePos3.xy + vec2(1.0, 1.0)) * 0.5;
        if (imagePos3.z >= 0.0 && imagePos3.z <= 1.0
            && texPos.x >= 0.0 && texPos.x <= 1.0
            && texPos.y >= 0.0 && texPos.y <= 1.0) {
            // https://stackoverflow.com/questions/72648980/opengl-sampler2d-array
            colorOut = texture(cameraImgs, vec3(1 - texPos.x, 1 - texPos.y, i));
            //int label = int(texture(semanticImgs, vec3(1 - texPos.x, 1 - texPos.y, i)).r * 255);
            //if (label != 2) // sea
            //    colorOut = colorOut / 10.0;
        }
    }
    
    p3d_FragColor = colorOut;
}