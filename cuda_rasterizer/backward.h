/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuda_fp16.h>
#include "half_struct.h"

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const __half* bg_color,
		const __half2* means2D,
		const half4* conic_opacity,
		const __half* colors,
		const __half* final_Ts,
		const uint32_t* n_contrib,
		const __half* dL_dpixels,
		half3* dL_dmean2D,
		half4* dL_dconic2D,
		__half* dL_dopacity,
		__half* dL_dcolors);

	void preprocess(
		int P, int D, int M,
		const half3* means,
		const int* radii,
		const __half* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const __half scale_modifier,
		const __half* cov3Ds,
		const __half* view,
		const __half* proj,
		const __half focal_x, __half focal_y,
		const __half tan_fovx, __half tan_fovy,
		const glm::vec3* campos,
		const half3* dL_dmean2D,
		const __half* dL_dconics,
		glm::vec3* dL_dmeans,
		__half* dL_dcolor,
		__half* dL_dcov3D,
		__half* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif