#include "config.h"
#include "stdio.h"
#include <cuda_fp16.h>
#pragma once

// #ifndef OPERATOR_OVERLOADING_H
// #define OPERATOR_OVERLOADING_H
inline __device__ __half operator*(__half a, float b){
    return a*__float2half(b);
}

inline __device__ __half operator*(float a, __half b){
    return __float2half(a)*b;
}

inline __device__ __half operator/(__half a, float b){
    return a/__float2half(b);
}

inline __device__ __half operator/(float a, __half b){
    return __float2half(a)/b;
}

inline __device__ __half operator+(__half a, float b){
    return a+__float2half(b);
}

inline __device__ __half operator+(float a, __half b){
    return __float2half(a)+b;
}

inline __device__ __half operator-(__half a, float b){
    return a+__float2half(b);
}

inline __device__ __half operator-(float a, __half b){
    return __float2half(a)+b;
}

inline __device__ bool operator>(__half a, float b){
    if(a>__float2half(b))
        return true;
    else
        return false;
}

inline __device__ bool operator<(__half a, float b){
    if(a<__float2half(b))
        return true;
    else
        return false;
}

inline __device__ bool operator<(float a, __half b){
    if(__float2half(a)<b)
        return true;
    else
        return false;
}

inline __device__ bool operator==(__half a, float b){
    if(a==__float2half(b))
        return true;
    else
        return false;
}

inline __device__ glm::vec3 operator*(__half a, glm::vec3 b){
    return __half2float(a)*b;
}

inline __device__ glm::vec3 operator*(glm::vec3 a, __half b){
    return a*__half2float(b);
}

inline __device__ __half min(__half a, __half b){
    if(a>b)
        return b;
    else
        return a;
}

inline __device__ __half max(__half a, __half b){
    if(a>b)
        return a;
    else
        return b;
}

inline __device__ __half operator!=(__half a, int b){
    if(a!=__int2half_rn(b))
        return true;
    else
        return false;
}

// #endif //OPERATOR_OVERLOADING_H