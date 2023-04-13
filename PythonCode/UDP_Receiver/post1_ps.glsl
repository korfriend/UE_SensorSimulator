#version 430

in vec2 l_texcoord0;
out vec4 p3d_FragColor;

uniform sampler2D texPass0;
uniform sampler2D texPass1;


void main()
{
    vec4 c = texture(texPass1, l_texcoord0);

    // To have a useless filter that outputs the original view
    // without changing anything, just use :
    //o_color  = c;

    // basic black and white effet
    float moyenne = (c.x + c.y + c.z)/3;
    //o_color = float4(moyenne, moyenne, moyenne, 1);
    p3d_FragColor = c;//vec4(l_texcoord0, 0, 1);
    //p3d_FragColor = vec4(l_texcoord0, 0, 1);
}