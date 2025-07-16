#pragma once
#include <cuda_runtime.h>
#include <math.h>

__host__ __device__ inline float2 make_float2(float s)
{
    return make_float2(s, s);
}

__host__ __device__ inline float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline float2 operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
__host__ __device__ inline float2 operator*(const float2& a, const float2& b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
__host__ __device__ inline float2 operator/(const float2& a, const float2& b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}

__host__ __device__ inline float2 operator+(const float2& a, float s)
{
    return make_float2(a.x + s, a.y + s);
}
__host__ __device__ inline float2 operator-(const float2& a, float s)
{
    return make_float2(a.x - s, a.y - s);
}
__host__ __device__ inline float2 operator*(const float2& a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
__host__ __device__ inline float2 operator/(const float2& a, float s)
{
    return make_float2(a.x / s, a.y / s);
}

__host__ __device__ inline float2 operator+(const float s, const float2& a) { return a + s; }
__host__ __device__ inline float2 operator-(const float s, const float2& a)
{
    return make_float2(s - a.x, s - a.y);
}
__host__ __device__ inline float2 operator*(const float s, const float2& a) { return a * s; }
__host__ __device__ inline float2 operator/(const float s, const float2& a)
{
    return make_float2(s / a.x, s / a.y);
}

__host__ __device__ inline float2 operator-(const float2& a)
{
    return make_float2(-a.x, -a.y);
}

__host__ __device__ inline float2& operator+=(float2& a, const float2& b)
{
    a.x += b.x; a.y += b.y; return a;
}
__host__ __device__ inline float2& operator-=(float2& a, const float2& b)
{
    a.x -= b.x; a.y -= b.y; return a;
}
__host__ __device__ inline float2& operator*=(float2& a, float s)
{
    a.x *= s; a.y *= s; return a;
}
__host__ __device__ inline float2& operator/=(float2& a, float s)
{
    a.x /= s; a.y /= s; return a;
}

__host__ __device__ inline bool operator==(const float2& a, const float2& b)
{
    return a.x == b.x && a.y == b.y;
}
__host__ __device__ inline bool operator!=(const float2& a, const float2& b)
{
    return !(a == b);
}

__host__ __device__ inline float dot(const float2& a, const float2& b)
{
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline float length2(const float2& v)
{
    return dot(v, v);
}

__host__ __device__ inline float length(const float2& v)
{
    return sqrt(length2(v));
}

__host__ __device__ inline float2 normalize(const float2& v)
{
    return v / length(v);
}


__host__ __device__ inline float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

__host__ __device__ inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ inline float3 operator/(const float3& a, const float3& b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ inline float3 operator+(const float3& a, float s)
{
    return make_float3(a.x + s, a.y + s, a.z + s);
}
__host__ __device__ inline float3 operator-(const float3& a, float s)
{
    return make_float3(a.x - s, a.y - s, a.z - s);
}
__host__ __device__ inline float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3 operator/(const float3& a, float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline float3 operator+(const float s, const float3& a) { return a + s; }
__host__ __device__ inline float3 operator-(const float s, const float3& a)
{
    return make_float3(s - a.x, s - a.y, s - a.z);
}
__host__ __device__ inline float3 operator*(const float s, const float3& a) { return a * s; }
__host__ __device__ inline float3 operator/(const float s, const float3& a)
{
    return make_float3(s / a.x, s / a.y, s / a.z);
}

__host__ __device__ inline float3 operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
__host__ __device__ inline float3& operator-=(float3& a, const float3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;
}
__host__ __device__ inline float3& operator*=(float3& a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; return a;
}
__host__ __device__ inline float3& operator/=(float3& a, float s)
{
    a.x /= s; a.y /= s; a.z /= s; return a;
}

__host__ __device__ inline bool operator==(const float3& a, const float3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
__host__ __device__ inline bool operator!=(const float3& a, const float3& b)
{
    return !(a == b);
}

__host__ __device__ inline float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length2(const float3& v)
{
    return dot(v, v);
}

__host__ __device__ inline float length(const float3& v)
{
    return sqrt(length2(v));
}

__host__ __device__ inline float3 normalize(const float3& v)
{
    return v / length(v);
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}


__host__ __device__ inline float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

__host__ __device__ inline float4 operator+(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ inline float4 operator-(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__host__ __device__ inline float4 operator*(const float4& a, const float4& b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__host__ __device__ inline float4 operator/(const float4& a, const float4& b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__host__ __device__ inline float4 operator+(const float4& a, float s)
{
    return make_float4(a.x + s, a.y + s, a.z + s, a.w + s);
}
__host__ __device__ inline float4 operator-(const float4& a, float s)
{
    return make_float4(a.x - s, a.y - s, a.z - s, a.w - s);
}
__host__ __device__ inline float4 operator*(const float4& a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__host__ __device__ inline float4 operator/(const float4& a, float s)
{
    return make_float4(a.x / s, a.y / s, a.z / s, a.w / s);
}

__host__ __device__ inline float4 operator+(const float s, const float4& a) { return a + s; }
__host__ __device__ inline float4 operator-(const float s, const float4& a)
{
    return make_float4(s - a.x, s - a.y, s - a.z, s - a.w);
}
__host__ __device__ inline float4 operator*(const float s, const float4& a) { return a * s; }
__host__ __device__ inline float4 operator/(const float s, const float4& a)
{
    return make_float4(s / a.x, s / a.y, s / a.z, s / a.w);
}

__host__ __device__ inline float4 operator-(const float4& a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

__host__ __device__ inline float4& operator+=(float4& a, const float4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a;
}
__host__ __device__ inline float4& operator-=(float4& a, const float4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a;
}
__host__ __device__ inline float4& operator*=(float4& a, float s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s; return a;
}
__host__ __device__ inline float4& operator/=(float4& a, float s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s; return a;
}

__host__ __device__ inline bool operator==(const float4& a, const float4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__host__ __device__ inline bool operator!=(const float4& a, const float4& b)
{
    return !(a == b);
}

__host__ __device__ inline float dot(const float4& a, const float4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ inline float length2(const float4& v)
{
    return dot(v, v);
}

__host__ __device__ inline float length(const float4& v)
{
    return sqrt(length2(v));
}

__host__ __device__ inline float4 normalize(const float4& v)
{
    return v / length(v);
}


__host__ __device__ inline int2 make_int2(int s)
{
    return make_int2(s, s);
}

__host__ __device__ inline int2 operator+(const int2& a, const int2& b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline int2 operator-(const int2& a, const int2& b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
__host__ __device__ inline int2 operator*(const int2& a, const int2& b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
__host__ __device__ inline int2 operator/(const int2& a, const int2& b)
{
    return make_int2(a.x / b.x, a.y / b.y);
}

__host__ __device__ inline int2 operator+(const int2& a, int s)
{
    return make_int2(a.x + s, a.y + s);
}
__host__ __device__ inline int2 operator-(const int2& a, int s)
{
    return make_int2(a.x - s, a.y - s);
}
__host__ __device__ inline int2 operator*(const int2& a, int s)
{
    return make_int2(a.x * s, a.y * s);
}
__host__ __device__ inline int2 operator/(const int2& a, int s)
{
    return make_int2(a.x / s, a.y / s);
}

__host__ __device__ inline int2 operator+(const int s, const int2& a) { return a + s; }
__host__ __device__ inline int2 operator-(const int s, const int2& a)
{
    return make_int2(s - a.x, s - a.y);
}
__host__ __device__ inline int2 operator*(const int s, const int2& a) { return a * s; }
__host__ __device__ inline int2 operator/(const int s, const int2& a)
{
    return make_int2(s / a.x, s / a.y);
}

__host__ __device__ inline int2 operator-(const int2& a)
{
    return make_int2(-a.x, -a.y);
}

__host__ __device__ inline int2& operator+=(int2& a, const int2& b)
{
    a.x += b.x; a.y += b.y; return a;
}
__host__ __device__ inline int2& operator-=(int2& a, const int2& b)
{
    a.x -= b.x; a.y -= b.y; return a;
}
__host__ __device__ inline int2& operator*=(int2& a, int s)
{
    a.x *= s; a.y *= s; return a;
}
__host__ __device__ inline int2& operator/=(int2& a, int s)
{
    a.x /= s; a.y /= s; return a;
}

__host__ __device__ inline bool operator==(const int2& a, const int2& b)
{
    return a.x == b.x && a.y == b.y;
}
__host__ __device__ inline bool operator!=(const int2& a, const int2& b)
{
    return !(a == b);
}


__host__ __device__ inline int3 make_int3(int s)
{
    return make_int3(s, s, s);
}

__host__ __device__ inline int3 operator+(const int3& a, const int3& b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline int3 operator-(const int3& a, const int3& b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline int3 operator*(const int3& a, const int3& b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ inline int3 operator/(const int3& a, const int3& b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ inline int3 operator+(const int3& a, int s)
{
    return make_int3(a.x + s, a.y + s, a.z + s);
}
__host__ __device__ inline int3 operator-(const int3& a, int s)
{
    return make_int3(a.x - s, a.y - s, a.z - s);
}
__host__ __device__ inline int3 operator*(const int3& a, int s)
{
    return make_int3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline int3 operator/(const int3& a, int s)
{
    return make_int3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline int3 operator+(const int s, const int3& a) { return a + s; }
__host__ __device__ inline int3 operator-(const int s, const int3& a)
{
    return make_int3(s - a.x, s - a.y, s - a.z);
}
__host__ __device__ inline int3 operator*(const int s, const int3& a) { return a * s; }
__host__ __device__ inline int3 operator/(const int s, const int3& a)
{
    return make_int3(s / a.x, s / a.y, s / a.z);
}

__host__ __device__ inline int3 operator-(const int3& a)
{
    return make_int3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline int3& operator+=(int3& a, const int3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
__host__ __device__ inline int3& operator-=(int3& a, const int3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;
}
__host__ __device__ inline int3& operator*=(int3& a, int s)
{
    a.x *= s; a.y *= s; a.z *= s; return a;
}
__host__ __device__ inline int3& operator/=(int3& a, int s)
{
    a.x /= s; a.y /= s; a.z /= s; return a;
}

__host__ __device__ inline bool operator==(const int3& a, const int3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
__host__ __device__ inline bool operator!=(const int3& a, const int3& b)
{
    return !(a == b);
}


__host__ __device__ inline int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}

__host__ __device__ inline int4 operator+(const int4& a, const int4& b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ inline int4 operator-(const int4& a, const int4& b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__host__ __device__ inline int4 operator*(const int4& a, const int4& b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__host__ __device__ inline int4 operator/(const int4& a, const int4& b)
{
    return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__host__ __device__ inline int4 operator+(const int4& a, int s)
{
    return make_int4(a.x + s, a.y + s, a.z + s, a.w + s);
}
__host__ __device__ inline int4 operator-(const int4& a, int s)
{
    return make_int4(a.x - s, a.y - s, a.z - s, a.w - s);
}
__host__ __device__ inline int4 operator*(const int4& a, int s)
{
    return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__host__ __device__ inline int4 operator/(const int4& a, int s)
{
    return make_int4(a.x / s, a.y / s, a.z / s, a.w / s);
}

__host__ __device__ inline int4 operator+(const int s, const int4& a) { return a + s; }
__host__ __device__ inline int4 operator-(const int s, const int4& a)
{
    return make_int4(s - a.x, s - a.y, s - a.z, s - a.w);
}
__host__ __device__ inline int4 operator*(const int s, const int4& a) { return a * s; }
__host__ __device__ inline int4 operator/(const int s, const int4& a)
{
    return make_int4(s / a.x, s / a.y, s / a.z, s / a.w);
}

__host__ __device__ inline int4 operator-(const int4& a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

__host__ __device__ inline int4& operator+=(int4& a, const int4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a;
}
__host__ __device__ inline int4& operator-=(int4& a, const int4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a;
}
__host__ __device__ inline int4& operator*=(int4& a, int s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s; return a;
}
__host__ __device__ inline int4& operator/=(int4& a, int s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s; return a;
}

__host__ __device__ inline bool operator==(const int4& a, const int4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__host__ __device__ inline bool operator!=(const int4& a, const int4& b)
{
    return !(a == b);
}


__host__ __device__ inline uint2 make_uint2(unsigned int s)
{
    return make_uint2(s, s);
}

__host__ __device__ inline uint2 operator+(const uint2& a, const uint2& b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline uint2 operator-(const uint2& a, const uint2& b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
__host__ __device__ inline uint2 operator*(const uint2& a, const uint2& b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
__host__ __device__ inline uint2 operator/(const uint2& a, const uint2& b)
{
    return make_uint2(a.x / b.x, a.y / b.y);
}

__host__ __device__ inline uint2 operator+(const uint2& a, unsigned int s)
{
    return make_uint2(a.x + s, a.y + s);
}
__host__ __device__ inline uint2 operator-(const uint2& a, unsigned int s)
{
    return make_uint2(a.x - s, a.y - s);
}
__host__ __device__ inline uint2 operator*(const uint2& a, unsigned int s)
{
    return make_uint2(a.x * s, a.y * s);
}
__host__ __device__ inline uint2 operator/(const uint2& a, unsigned int s)
{
    return make_uint2(a.x / s, a.y / s);
}

__host__ __device__ inline uint2 operator+(const unsigned int s, const uint2& a) { return a + s; }
__host__ __device__ inline uint2 operator-(const unsigned int s, const uint2& a)
{
    return make_uint2(s - a.x, s - a.y);
}
__host__ __device__ inline uint2 operator*(const unsigned int s, const uint2& a) { return a * s; }
__host__ __device__ inline uint2 operator/(const unsigned int s, const uint2& a)
{
    return make_uint2(s / a.x, s / a.y);
}

__host__ __device__ inline uint2 operator-(const uint2& a)
{
    return make_uint2(-a.x, -a.y);
}

__host__ __device__ inline uint2& operator+=(uint2& a, const uint2& b)
{
    a.x += b.x; a.y += b.y; return a;
}
__host__ __device__ inline uint2& operator-=(uint2& a, const uint2& b)
{
    a.x -= b.x; a.y -= b.y; return a;
}
__host__ __device__ inline uint2& operator*=(uint2& a, unsigned int s)
{
    a.x *= s; a.y *= s; return a;
}
__host__ __device__ inline uint2& operator/=(uint2& a, unsigned int s)
{
    a.x /= s; a.y /= s; return a;
}

__host__ __device__ inline bool operator==(const uint2& a, const uint2& b)
{
    return a.x == b.x && a.y == b.y;
}
__host__ __device__ inline bool operator!=(const uint2& a, const uint2& b)
{
    return !(a == b);
}


__host__ __device__ inline uint3 make_uint3(unsigned int s)
{
    return make_uint3(s, s, s);
}

__host__ __device__ inline uint3 operator+(const uint3& a, const uint3& b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline uint3 operator-(const uint3& a, const uint3& b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline uint3 operator*(const uint3& a, const uint3& b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ inline uint3 operator/(const uint3& a, const uint3& b)
{
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ inline uint3 operator+(const uint3& a, unsigned int s)
{
    return make_uint3(a.x + s, a.y + s, a.z + s);
}
__host__ __device__ inline uint3 operator-(const uint3& a, unsigned int s)
{
    return make_uint3(a.x - s, a.y - s, a.z - s);
}
__host__ __device__ inline uint3 operator*(const uint3& a, unsigned int s)
{
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline uint3 operator/(const uint3& a, unsigned int s)
{
    return make_uint3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline uint3 operator+(const unsigned int s, const uint3& a) { return a + s; }
__host__ __device__ inline uint3 operator-(const unsigned int s, const uint3& a)
{
    return make_uint3(s - a.x, s - a.y, s - a.z);
}
__host__ __device__ inline uint3 operator*(const unsigned int s, const uint3& a) { return a * s; }
__host__ __device__ inline uint3 operator/(const unsigned int s, const uint3& a)
{
    return make_uint3(s / a.x, s / a.y, s / a.z);
}

__host__ __device__ inline uint3 operator-(const uint3& a)
{
    return make_uint3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline uint3& operator+=(uint3& a, const uint3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
__host__ __device__ inline uint3& operator-=(uint3& a, const uint3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;
}
__host__ __device__ inline uint3& operator*=(uint3& a, unsigned int s)
{
    a.x *= s; a.y *= s; a.z *= s; return a;
}
__host__ __device__ inline uint3& operator/=(uint3& a, unsigned int s)
{
    a.x /= s; a.y /= s; a.z /= s; return a;
}

__host__ __device__ inline bool operator==(const uint3& a, const uint3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
__host__ __device__ inline bool operator!=(const uint3& a, const uint3& b)
{
    return !(a == b);
}


__host__ __device__ inline uint4 make_uint4(unsigned int s)
{
    return make_uint4(s, s, s, s);
}

__host__ __device__ inline uint4 operator+(const uint4& a, const uint4& b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ inline uint4 operator-(const uint4& a, const uint4& b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__host__ __device__ inline uint4 operator*(const uint4& a, const uint4& b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__host__ __device__ inline uint4 operator/(const uint4& a, const uint4& b)
{
    return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__host__ __device__ inline uint4 operator+(const uint4& a, unsigned int s)
{
    return make_uint4(a.x + s, a.y + s, a.z + s, a.w + s);
}
__host__ __device__ inline uint4 operator-(const uint4& a, unsigned int s)
{
    return make_uint4(a.x - s, a.y - s, a.z - s, a.w - s);
}
__host__ __device__ inline uint4 operator*(const uint4& a, unsigned int s)
{
    return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__host__ __device__ inline uint4 operator/(const uint4& a, unsigned int s)
{
    return make_uint4(a.x / s, a.y / s, a.z / s, a.w / s);
}

__host__ __device__ inline uint4 operator+(const unsigned int s, const uint4& a) { return a + s; }
__host__ __device__ inline uint4 operator-(const unsigned int s, const uint4& a)
{
    return make_uint4(s - a.x, s - a.y, s - a.z, s - a.w);
}
__host__ __device__ inline uint4 operator*(const unsigned int s, const uint4& a) { return a * s; }
__host__ __device__ inline uint4 operator/(const unsigned int s, const uint4& a)
{
    return make_uint4(s / a.x, s / a.y, s / a.z, s / a.w);
}

__host__ __device__ inline uint4 operator-(const uint4& a)
{
    return make_uint4(-a.x, -a.y, -a.z, -a.w);
}

__host__ __device__ inline uint4& operator+=(uint4& a, const uint4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a;
}
__host__ __device__ inline uint4& operator-=(uint4& a, const uint4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a;
}
__host__ __device__ inline uint4& operator*=(uint4& a, unsigned int s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s; return a;
}
__host__ __device__ inline uint4& operator/=(uint4& a, unsigned int s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s; return a;
}

__host__ __device__ inline bool operator==(const uint4& a, const uint4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__host__ __device__ inline bool operator!=(const uint4& a, const uint4& b)
{
    return !(a == b);
}


__host__ __device__ inline double2 make_double2(double s)
{
    return make_double2(s, s);
}

__host__ __device__ inline double2 operator+(const double2& a, const double2& b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline double2 operator-(const double2& a, const double2& b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}
__host__ __device__ inline double2 operator*(const double2& a, const double2& b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}
__host__ __device__ inline double2 operator/(const double2& a, const double2& b)
{
    return make_double2(a.x / b.x, a.y / b.y);
}

__host__ __device__ inline double2 operator+(const double2& a, double s)
{
    return make_double2(a.x + s, a.y + s);
}
__host__ __device__ inline double2 operator-(const double2& a, double s)
{
    return make_double2(a.x - s, a.y - s);
}
__host__ __device__ inline double2 operator*(const double2& a, double s)
{
    return make_double2(a.x * s, a.y * s);
}
__host__ __device__ inline double2 operator/(const double2& a, double s)
{
    return make_double2(a.x / s, a.y / s);
}

__host__ __device__ inline double2 operator+(const double s, const double2& a) { return a + s; }
__host__ __device__ inline double2 operator-(const double s, const double2& a)
{
    return make_double2(s - a.x, s - a.y);
}
__host__ __device__ inline double2 operator*(const double s, const double2& a) { return a * s; }
__host__ __device__ inline double2 operator/(const double s, const double2& a)
{
    return make_double2(s / a.x, s / a.y);
}

__host__ __device__ inline double2 operator-(const double2& a)
{
    return make_double2(-a.x, -a.y);
}

__host__ __device__ inline double2& operator+=(double2& a, const double2& b)
{
    a.x += b.x; a.y += b.y; return a;
}
__host__ __device__ inline double2& operator-=(double2& a, const double2& b)
{
    a.x -= b.x; a.y -= b.y; return a;
}
__host__ __device__ inline double2& operator*=(double2& a, double s)
{
    a.x *= s; a.y *= s; return a;
}
__host__ __device__ inline double2& operator/=(double2& a, double s)
{
    a.x /= s; a.y /= s; return a;
}

__host__ __device__ inline bool operator==(const double2& a, const double2& b)
{
    return a.x == b.x && a.y == b.y;
}
__host__ __device__ inline bool operator!=(const double2& a, const double2& b)
{
    return !(a == b);
}

__host__ __device__ inline double dot(const double2& a, const double2& b)
{
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline double length2(const double2& v)
{
    return dot(v, v);
}

__host__ __device__ inline double length(const double2& v)
{
    return sqrtf(length2(v));
}

__host__ __device__ inline double2 normalize(const double2& v)
{
    return v / length(v);
}


__host__ __device__ inline double3 make_double3(double s)
{
    return make_double3(s, s, s);
}

__host__ __device__ inline double3 operator+(const double3& a, const double3& b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline double3 operator-(const double3& a, const double3& b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline double3 operator*(const double3& a, const double3& b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__host__ __device__ inline double3 operator/(const double3& a, const double3& b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ inline double3 operator+(const double3& a, double s)
{
    return make_double3(a.x + s, a.y + s, a.z + s);
}
__host__ __device__ inline double3 operator-(const double3& a, double s)
{
    return make_double3(a.x - s, a.y - s, a.z - s);
}
__host__ __device__ inline double3 operator*(const double3& a, double s)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline double3 operator/(const double3& a, double s)
{
    return make_double3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline double3 operator+(const double s, const double3& a) { return a + s; }
__host__ __device__ inline double3 operator-(const double s, const double3& a)
{
    return make_double3(s - a.x, s - a.y, s - a.z);
}
__host__ __device__ inline double3 operator*(const double s, const double3& a) { return a * s; }
__host__ __device__ inline double3 operator/(const double s, const double3& a)
{
    return make_double3(s / a.x, s / a.y, s / a.z);
}

__host__ __device__ inline double3 operator-(const double3& a)
{
    return make_double3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline double3& operator+=(double3& a, const double3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
__host__ __device__ inline double3& operator-=(double3& a, const double3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;
}
__host__ __device__ inline double3& operator*=(double3& a, double s)
{
    a.x *= s; a.y *= s; a.z *= s; return a;
}
__host__ __device__ inline double3& operator/=(double3& a, double s)
{
    a.x /= s; a.y /= s; a.z /= s; return a;
}

__host__ __device__ inline bool operator==(const double3& a, const double3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
__host__ __device__ inline bool operator!=(const double3& a, const double3& b)
{
    return !(a == b);
}

__host__ __device__ inline double dot(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline double length2(const double3& v)
{
    return dot(v, v);
}

__host__ __device__ inline double length(const double3& v)
{
    return sqrtf(length2(v));
}

__host__ __device__ inline double3 normalize(const double3& v)
{
    return v / length(v);
}


__host__ __device__ inline double4 make_double4(double s)
{
    return make_double4(s, s, s, s);
}

__host__ __device__ inline double4 operator+(const double4& a, const double4& b)
{
    return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ inline double4 operator-(const double4& a, const double4& b)
{
    return make_double4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__host__ __device__ inline double4 operator*(const double4& a, const double4& b)
{
    return make_double4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__host__ __device__ inline double4 operator/(const double4& a, const double4& b)
{
    return make_double4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__host__ __device__ inline double4 operator+(const double4& a, double s)
{
    return make_double4(a.x + s, a.y + s, a.z + s, a.w + s);
}
__host__ __device__ inline double4 operator-(const double4& a, double s)
{
    return make_double4(a.x - s, a.y - s, a.z - s, a.w - s);
}
__host__ __device__ inline double4 operator*(const double4& a, double s)
{
    return make_double4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__host__ __device__ inline double4 operator/(const double4& a, double s)
{
    return make_double4(a.x / s, a.y / s, a.z / s, a.w / s);
}

__host__ __device__ inline double4 operator+(const double s, const double4& a) { return a + s; }
__host__ __device__ inline double4 operator-(const double s, const double4& a)
{
    return make_double4(s - a.x, s - a.y, s - a.z, s - a.w);
}
__host__ __device__ inline double4 operator*(const double s, const double4& a) { return a * s; }
__host__ __device__ inline double4 operator/(const double s, const double4& a)
{
    return make_double4(s / a.x, s / a.y, s / a.z, s / a.w);
}

__host__ __device__ inline double4 operator-(const double4& a)
{
    return make_double4(-a.x, -a.y, -a.z, -a.w);
}

__host__ __device__ inline double4& operator+=(double4& a, const double4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a;
}
__host__ __device__ inline double4& operator-=(double4& a, const double4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a;
}
__host__ __device__ inline double4& operator*=(double4& a, double s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s; return a;
}
__host__ __device__ inline double4& operator/=(double4& a, double s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s; return a;
}

__host__ __device__ inline bool operator==(const double4& a, const double4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__host__ __device__ inline bool operator!=(const double4& a, const double4& b)
{
    return !(a == b);
}

__host__ __device__ inline double dot(const double4& a, const double4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ inline double length2(const double4& v)
{
    return dot(v, v);
}

__host__ __device__ inline double length(const double4& v)
{
    return sqrtf(length2(v));
}

__host__ __device__ inline double4 normalize(const double4& v)
{
    return v / length(v);
}
