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


const float K1_ = 1.281584985127447;
const float K2_ = 0.170043067138006;
const float K3_ = -0.023341058557079;
const float K4_ = 0.007690791651144;
const float K5_ = -0.001380968639013;


const float img_width_ = 1920;
const float img_height_ = 1080;

const float fx_ = 345.12136354806347;
const float fy_ = 346.09009197978003;
const float cx_ = 959.5;// - img_width_ / 2;
const float cy_ = 539.5;// - img_height_ / 2;

vec2 distortPoint(vec2 Array_uv)
{
        float position_x = Array_uv.x * img_width_;
        float position_y = Array_uv.y * img_height_;

        position_x = (position_x - cx_) / fx_;
        position_y = (position_y - cy_) / fy_;

        const float rCam = sqrt(position_x * position_x + position_y * position_y);
        float theta = atan(rCam);
        //const float phi = atan2(position_y, position_x);
        const float phi = atan(position_y, position_x);

        const float theta3 = pow(theta, 3);
        const float theta5 = pow(theta, 5);
        const float theta7 = pow(theta, 7);
        const float theta9 = pow(theta, 9);

        const float radial_distance =
            K1_ * theta + K2_ * theta3 + K3_ * theta5 + K4_ * theta7 + K5_ * theta9;
        position_x = radial_distance * cos(phi) * fx_ + cx_;
        position_y = radial_distance * sin(phi) * fy_ + cy_;

        Array_uv.x = position_x;
        Array_uv.y = position_y;
        return vec2(Array_uv.x / img_width_, Array_uv.y / img_height_);
}

void main() {
    vec3 pos = worldcoord;
    mat4 viewProjs[4] = {matViewProj0, matViewProj1, matViewProj2, matViewProj3};
    int count = 0;
    int mapProp = 0;    // 0 : undef, 1 : ground, 2 : target geometry
    int overlapIndex[4] = {-1, -1, -1, -1};
    for (int i = 0; i < 4; i++) 
    {
        vec4 imagePos = viewProjs[i] * vec4(pos, 1.0);
        if (imagePos.w <= 0.001) imagePos.w = 0.001;
        vec3 imagePos3 = imagePos.xyz / imagePos.w;
        vec2 texPos = (imagePos3.xy + vec2(1.0, 1.0)) * 0.5;
        texPos = distortPoint(texPos);
        if (imagePos3.z >= 0.0 && imagePos3.z <= 1.0
            && texPos.x >= 0.0 && texPos.x <= 1.0
            && texPos.y >= 0.0 && texPos.y <= 1.0) {
                overlapIndex[count++] = i;
                //int semantic = int(texture(semanticImgs, vec3(1 - texPos.x, 1 - texPos.y, i)).r * 255.0 + 0.5);
                
                const ivec2 texIdx2d = ivec2((texPos.x) * img_width_ + 0.5, (1 - texPos.y) * img_height_ + 0.5);
                int semantic = texelFetch(semanticImgs, ivec3(texIdx2d, i), 0).r;
                //if(semantic == 0) semantic = 1;
                if (semantic > mapProp / 100) mapProp = semantic * 100 + i;
        }
    }

    //p3d_FragColor = vec4(0, 0, 1, 1);
    
    frag_color0 = uvec4(mapProp, overlapIndex[0], overlapIndex[1], count);
    //frag_color0 = uvec4(mapProp, floatBitsToUint(texcoord.x), floatBitsToUint(texcoord.y), count);
    frag_color1 = uvec4(floatBitsToUint(pos), 0);
}
