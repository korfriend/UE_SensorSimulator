#version 430
precision highp float;
precision highp int;

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
    const float phi = atan(position_y, position_x);

    const float theta3 = pow(theta, 3);
    const float theta5 = pow(theta, 5);
    const float theta7 = pow(theta, 7);
    const float theta9 = pow(theta, 9);

    const float radial_distance =
        K1_ * theta + K2_ * theta3 + K3_ * theta5 + K4_ * theta7 + K5_ * theta9;
    position_x = radial_distance * cos(phi) * fx_ + cx_;
    position_y = radial_distance * sin(phi) * fy_ + cy_;

    vec2 uv_out;
    uv_out.x = position_x / img_width_;
    uv_out.y = position_y / img_height_;
    return uv_out;//vec2(Array_uv.x / img_width_, Array_uv.y / img_height_);
}

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

    // height correction
    switch (semantic / 100) {
        case 1: pos.z = -50; break;
        case 2: pos.z = -50; break;
    }

    int overlapIndex[4] = {int(camId0), int(camId1), 0, 0};
    semantic = 0;
    for (int i = 0; i < count; i++) {
        int camId = overlapIndex[i];
        vec4 imagePos = viewProjs[camId] * vec4(pos, 1.0);
        vec2 imagePos2D = imagePos.xy / imagePos.w;
        vec2 texPos = (imagePos2D + vec2(1.0, 1.0)) * 0.5;
        texPos = distortPoint(texPos);

        ivec2 texIdx2d = ivec2((texPos.x) * img_width_ + 0.5, (1 - texPos.y) * img_height_ + 0.5);
        semantic = max(semantic, texelFetch(semanticImgs, ivec3(texIdx2d, camId), 0).r);
    }

    switch (semantic) {
        case 1: colorOut += vec4(0, 0, 0.2, 1); break;
        case 2: colorOut += vec4(0.2, 0, 0, 1); break;
    }

    if (sceneColor.r + sceneColor.g + sceneColor.b > 0)
        colorOut = sceneColor;

    p3d_FragColor = colorOut;
}