#include <cuda_fp16.h>
#pragma once
struct half3
{
    /* data */
    __half x;
    __half y;
    __half z;
};

struct half4
{
    /* data */
    __half x;
    __half y;
    __half z;
    __half w;
};

