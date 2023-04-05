#version 430

in vec2 l_texcoord0;
out vec4 p3d_FragColor;

uniform sampler2D k_tex;


void main()
{
    vec4 c = texture(k_tex, l_texcoord0);

    // To have a useless filter that outputs the original view
    // without changing anything, just use :
    //o_color  = c;

    // basic black and white effet
    float moyenne = (c.x + c.y + c.z)/3;
    //o_color = float4(moyenne, moyenne, moyenne, 1);
    p3d_FragColor = c;//vec4(l_texcoord0, 0, 1);
}