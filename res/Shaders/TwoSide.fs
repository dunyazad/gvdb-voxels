#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos;

uniform vec3 cameraPos;
uniform int useSolidColor;
uniform vec3 solidColor;

out vec4 FragColor;

void main()
{
    // gl_FrontFacing: true if front face, false if back face
    vec3 normal = vNormal;
    if (!gl_FrontFacing)
    {
        normal = -normal; // Flip normal for back faces
    }

    vec3 lightDir = normalize(cameraPos - vFragPos);
    float lighting = max(dot(normalize(normal), lightDir), 0.2);

    vec4 baseColor;
    if (useSolidColor == 0)
    {
        baseColor = vColor;
    }
    else
    {
        baseColor = vec4(solidColor, 1.0);
    }
    FragColor = baseColor * lighting;
}
