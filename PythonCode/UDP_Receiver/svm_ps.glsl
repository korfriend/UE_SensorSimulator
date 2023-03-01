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
//uniform vec4 testInts[4];

uniform mat4 matTest0;
uniform mat4 matTest1;

uniform sampler2DArray cameraImgs;

void main() {
    vec3 pos = worldcoord;
    mat4 viewProjs[4] = {matViewProj0, matViewProj1, matViewProj2, matViewProj3};

    vec4 colorOut = vec4(0); 
#ifdef __DEBUG
    vec4 imagePos = matViewProj0 * vec4(pos, 1.0);
    vec3 imagePos3 = imagePos.xyz / imagePos.w;
    vec2 texPos = (imagePos3.xy + vec2(1.0, 1.0)) * 0.5;
    if(imagePos3.z < 0)
        colorOut = vec4(0.01f, 0.0f, 0.0f, 1.0f);
    else if(imagePos3.z > 1)
        colorOut = vec4(0.5, 0, 1, 1);
    else if(texPos.x < 0)
        colorOut = vec4(1, 0, 0, 1);
    else if(texPos.y < 0)
        colorOut = vec4(0, 1, 0, 1);
    else if(texPos.x > 1)
        colorOut = vec4(0, 0, 1, 1);
    else if(texPos.y > 1)
        colorOut = vec4(1, 1, 0, 1);
    else {
        colorOut = texture(cameraImgs, vec3(1 - texPos.x, 1 - texPos.y, 0));//vec4(texPos, 0, 1);
    }
#else
    for (int i = 0; i < 4; i++) {
        vec4 imagePos = viewProjs[i] * vec4(pos, 1.0);
        vec3 imagePos3 = imagePos.xyz / imagePos.w;
        vec2 texPos = (imagePos3.xy + vec2(1.0, 1.0)) * 0.5;
        if (imagePos3.z >= 0.0 && imagePos3.z <= 1.0
            && texPos.x >= 0.0 && texPos.x <= 1.0
            && texPos.y >= 0.0 && texPos.y <= 1.0) {
            colorOut = texture(cameraImgs, vec3(1 - texPos.x, 1 - texPos.y, i));
        }
    }
#endif
    /**/
    p3d_FragColor = colorOut;
    /*
    sampler2D texs[4] = {myTexture0, myTexture1, myTexture2, myTexture3}
    vec4 colors[4];
    for(int i = 0; i < 4; i++) {
        vec4 imagePos = world2image * vec4(worldcoord, 1.0);
        imagePos.xyz /= imagePos.w;
        vec2 texPos = (imagePos.xy + vec2(1.0, 1.0)) * 0.5;
        
        colors[i] = texture(texs[i], texPos0);
    }

    vec4 color1 = texture(myTexture0, texPos1);
    vec4 color2 = texture(myTexture0, texPos2);
    vec4 color3 = texture(myTexture0, texPos3);

    float count = 0;
    if(color0.a > 0) 
        count
    /**/

    //vec4 colorSum = color0 + color1 + color2 + color3 / count;





    //vec4 color = texture(myTexture0, texPos);
    //p3d_FragColor = vec4(texPos.xy, 1, 1); color.rgba;
    //p3d_FragColor = texture(p3d_Texture0, texcoord);
    //p3d_FragColor = vec4(texcoord.x, texcoord.y, 0.0, 1.0);
    //p3d_FragColor = vec4(1, 0, 0.0, 1.0);
}

void main2() {
    //mat4 w2iT = transpose(matTest1); 
    //mat4 view = (matTest0);
    //mat4 vp = world2image0;
    vec3 pos = worldcoord;
    //pos.z = -500;
    vec4 imagePos = matViewProj0 * vec4(pos, 1.0);
    vec3 imagePos3 = imagePos.xyz / imagePos.w;
    //vec4 imagePosV = matTest0 * vec4(pos, 1.0);
    //vec3 imagePosV3 = imagePosV.xyz / imagePosV.w;
    vec2 texPos = (imagePos3.xy + vec2(1.0, 1.0)) * 0.5;
    vec4 colorOut = vec4(0);
    if(imagePos3.z < 0)
        colorOut = vec4(0.01f, 0.0f, 0.0f, 1.0f);
    else if(imagePos3.z > 1)
        colorOut = vec4(0.5, 0, 1, 1);
    else if(texPos.x < 0)
        colorOut = vec4(1, 0, 0, 1);
    else if(texPos.y < 0)
        colorOut = vec4(0, 1, 0, 1);
    else if(texPos.x > 1)
        colorOut = vec4(0, 0, 1, 1);
    else if(texPos.y > 1)
        colorOut = vec4(1, 1, 0, 1);
    else {
        //colorOut = texture(myTexture0, texPos);//vec4(texPos, 0, 1);
        colorOut = texture(cameraImgs, vec3(1 - texPos.x, 1 - texPos.y, 0));//vec4(texPos, 0, 1);
        //colorOut = vec4(texPos.x, 1 - texPos.y, 0, 1);
    }
    /**/
    p3d_FragColor = colorOut;
    /*
    sampler2D texs[4] = {myTexture0, myTexture1, myTexture2, myTexture3}
    vec4 colors[4];
    for(int i = 0; i < 4; i++) {
        vec4 imagePos = world2image * vec4(worldcoord, 1.0);
        imagePos.xyz /= imagePos.w;
        vec2 texPos = (imagePos.xy + vec2(1.0, 1.0)) * 0.5;
        
        colors[i] = texture(texs[i], texPos0);
    }

    vec4 color1 = texture(myTexture0, texPos1);
    vec4 color2 = texture(myTexture0, texPos2);
    vec4 color3 = texture(myTexture0, texPos3);

    float count = 0;
    if(color0.a > 0) 
        count
    /**/

    //vec4 colorSum = color0 + color1 + color2 + color3 / count;

    //vec4 color = texture(myTexture0, texPos);
    //p3d_FragColor = vec4(texPos.xy, 1, 1); color.rgba;
    //p3d_FragColor = texture(p3d_Texture0, texcoord);
    //p3d_FragColor = vec4(texcoord.x, texcoord.y, 0.0, 1.0);
    //p3d_FragColor = vec4(1, 0, 0.0, 1.0);
}