#ifndef CUDA_RENDER_H_INCLUDED
#define CUDA_RENDER_H_INCLUDED

#include <vector>
#include <functional>
#include <cuda_runtime_api.h>


int RenderForwardCUDA(
	const int P, const int S, const int width, const int height,
	const float* depths_to_sort,
	const float* t_A_inv,
	const float* t_features,
	const float* t_depths,
	const float* t_sdfs_var,
	const int* t_proj_2D_min_max,
	const int32_t* rect,
	const int32_t* tiles_touched,
	const bool* visibility_filter,
	std::function<char* (size_t)> binningBuffer,
	int* contrib,
	float* alpha,
	float* feature,
	float* error_map,
	int2* ranges,
	float alpha_threshold,
	bool debug);

void RenderBackwardCUDA(
	const int P, const int S, const int num_rendered,
	const int width, const int height,
	const float* t_A_inv,
	const float* t_features,
	const float* t_depths,
	const float* t_sdfs_var,
	const int* t_proj_2D_min_max,
	const float* alpha,
	const float* feature,
	const int* contrib,
	const int2* ranges,
	char* binning_buffer,
	const float* dL_dalpha,
	const float* dL_dfeature,
	float* dL_dt_A_inv,
	float* dL_dfeatures,
	float* dL_dt_depths,
	float* dL_dt_sdfs_var,
	float alpha_threshold,
	bool debug);

#endif