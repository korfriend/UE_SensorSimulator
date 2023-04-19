#version 430
//precision highp float;
//precision highp int;
//#extension GL_EXT_texture_array
//#extension GL_NV_texture_array
const float PI = 3.14159265359;
// fragment-pixel output color
//out vec4 p3d_FragColor;

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
uniform isampler2DArray semanticImgs;

out uvec4 frag_color0;
layout(location = 1) out uvec4 frag_color1;

void main() {
    vec3 pos = worldcoord;
    mat4 viewProjs[4] = {matViewProj0, matViewProj1, matViewProj2, matViewProj3};

    int count = 0;
    int mapProp = 0;    // 0 : undef, 1 : ground, 2 : target geometry
    int overlapIndex[4] = {-1, -1, -1, -1};
    for (int i = 0; i < 4; i++) 
    {
        vec4 imagePos = viewProjs[i] * vec4(pos, 1.0);
        vec3 imagePos3 = imagePos.xyz / imagePos.w;
        vec2 texPos = (imagePos3.xy + vec2(1.0, 1.0)) * 0.5;
        if (imagePos3.z >= 0.0 && imagePos3.z <= 1.0
            && texPos.x >= 0.0 && texPos.x <= 1.0
            && texPos.y >= 0.0 && texPos.y <= 1.0) {
                overlapIndex[count++] = i;
                //int semantic = int(texture(semanticImgs, vec3(1 - texPos.x, 1 - texPos.y, i)).r * 255.0 + 0.5);
                
                const ivec2 texIdx2d = ivec2((1 - texPos.x) * 500 + 0.5, (1 - texPos.y) * 300 + 0.5);
                int semantic = texelFetch(semanticImgs, ivec3(texIdx2d, i), 0).r;
                if (semantic > mapProp) mapProp = semantic;
        }
    }

    //p3d_FragColor = vec4(0, 0, 1, 1);
    
    frag_color0 = uvec4(mapProp, overlapIndex[0], overlapIndex[1], count);
    frag_color1 = uvec4(floatBitsToUint(pos), 0);
}