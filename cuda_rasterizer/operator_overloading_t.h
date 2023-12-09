#include "config.h"
#include "stdio.h"
#include <cuda_fp16.h>

// #ifndef OPERATOR_OVERLOADING_H
// #define OPERATOR_OVERLOADING_H
__device__ __half operator*(__half a, float b);

__device__ __half operator*(float a, __half b);

__device__ __half operator/(__half a, float b);

__device__ __half operator/(float a, __half b);

__device__ __half operator+(__half a, float b);

__device__ __half operator+(float a, __half b);

__device__ __half operator-(__half a, float b);

__device__ __half operator-(float a, __half b);

__device__ bool operator>(__half a, float b);

__device__ bool operator<(__half a, float b);

__device__ bool operator<(float a, __half b);

__device__ bool operator==(__half a, float b);

__device__ glm::vec3 operator*(__half a, glm::vec3 b);

__device__ glm::vec3 operator*(glm::vec3 a, __half b);

__device__ __half min(__half a, __half b);

__device__ __half max(__half a, __half b);

__device__ __half operator!=(__half a, int b);

// #endif //OPERATOR_OVERLOADING_H