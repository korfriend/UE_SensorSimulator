#version 430
//#extension GL_EXT_texture_array
//#extension GL_NV_texture_array

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

uniform mat4 matTest0;
uniform mat4 matTest1;

uniform sampler2DArray cameraImgs;
uniform sampler2DArray semanticImgs;

void main() {
    vec3 pos = worldcoord;
    mat4 viewProjs[4] = {matViewProj0, matViewProj1, matViewProj2, matViewProj3};

    vec4 colorOut = vec4(0); 
    
    int count = 0;
    int overlapIndex[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        vec4 imagePos = viewProjs[i] * vec4(pos, 1.0);
        vec3 imagePos3 = imagePos.xyz / imagePos.w;
        vec2 texPos = (imagePos3.xy + vec2(1.0, 1.0)) * 0.5;
        if (imagePos3.z >= 0.0 && imagePos3.z <= 1.0
            && texPos.x >= 0.0 && texPos.x <= 1.0
            && texPos.y >= 0.0 && texPos.y <= 1.0) {
                overlapIndex[count++] = i;
        }
    }

    switch (count) {
        case 1: {
            switch(overlapIndex[0]) {
                case 0: colorOut = vec4(1, 0, 0, 1); break;
                case 1: colorOut = vec4(0, 1, 0, 1); break;
                case 2: colorOut = vec4(0, 0, 1, 1); break;
                case 3: colorOut = vec4(1, 1, 1, 1); break;
            }
            break;
        }
        case 2: {
            int idx0 = overlapIndex[0];
            int idx1 = overlapIndex[1];
            vec3 camPos0 = camPositions[idx0].xyz;
            vec3 camPos1 = camPositions[idx1].xyz;

            vec2 vecX0 = normalize(pos.xz - camPos0.xz);
            vec2 vecX1 = normalize(pos.xz - camPos1.xz);
            vec2 vecY0 = normalize(pos.yz - camPos1.yz);
            vec2 vecY1 = normalize(pos.yz - camPos1.yz);
            float angleX = acos(dot(vecX0, vecY0));
            float angleY = acos(dot(vecX1, vecY1));

            int enc = idx0 * 2 + idx1;
            switch(enc) {
                case 1: colorOut = vec4(1, 1, 0, 1); break;
                case 4: colorOut = vec4(0, 1, 1, 1); break;
                case 7: colorOut = vec4(0.5, 0.5, 1, 1); break;
                case 3: colorOut = vec4(1, 0.5, 0.5, 1); break;
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