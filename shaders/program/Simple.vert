/*
--------------------------------------------------------------------------------

	Revelation Shaders

	Copyright (C) 2026 HaringPro
	Apache License 2.0

--------------------------------------------------------------------------------
*/

//======// Utility //=============================================================================//

#include "/lib/Utility.glsl"

//======// Uniform //=============================================================================//

uniform vec2 renderScale;

//======// Main //================================================================================//
void main() {
	// Full screen triangle
    // http://www.altdev.co/2011/08/08/interesting-vertex-shader-trick/
    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    #if ((RENDER_SCALE_1000X != 1000) || SR_ENABLE) && !defined NO_SCALE
        uv *= renderScale;
    #endif
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
