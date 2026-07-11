//
//  main.metal
//  Grimlet
//
//  Created by Roddy MacCrimmon on 2026-07-11.
//

#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float2 texCoord;
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vtx(const device Vertex *v [[buffer(0)]], uint id [[vertex_id]]) {
    VertexOut o;
    o.position = float4(v[id].position, 0, 1);
    o.texCoord = v[id].texCoord;
    return o;
}

fragment float4 frag(VertexOut i [[stage_in]], texture2d<float> t [[texture(0)]]) {
    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    return t.sample(s, i.texCoord);
}
