#version 430

// fragment-pixel output color
out vec4 p3d_FragColor;

// input from vertex shader
in vec2 texcoord;
in vec3 worldcoord;

uniform mat4 world2image0;
//uniform mat4 world2image1;
//uniform mat4 world2image2;
//uniform mat4 world2image3;

uniform sampler2D p3d_Texture0;
uniform sampler2D myTexture0;
uniform sampler2D myTexture1;
uniform sampler2D myTexture2;
uniform sampler2D myTexture3;

void main() {
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





    vec4 color = texture(myTexture0, texcoord);
    p3d_FragColor = color.rgba;
    //p3d_FragColor = texture(p3d_Texture0, texcoord);
    //p3d_FragColor = vec4(texcoord.x, texcoord.y, 0.0, 1.0);
    //p3d_FragColor = vec4(1, 0, 0.0, 1.0);
}