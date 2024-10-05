#ifndef CUDA_PREPROCESS_H_INCLUDED
#define CUDA_PREPROCESS_H_INCLUDED

#include <vector>
#include <functional>
#include <cuda_runtime_api.h>


void PreprocessForwardCUDA(
	const int P, const int V, const int H, const int W,
	const float* vertices,
	const int* tet_indices,
	const float* proj,
	const float* w2c,
    float* depths,
    float* v_2Ds,
	float* t_A_inv,
	float* depths_to_sort,
	int* t_proj_2D_min_max,
	int* rect,
	int* tiles_touched,
	bool* visibility_filter
);

void PreprocessBackwardCUDA(
	const int P, const int V, const int H, const int W,
	const float* vertices,
	const int* tet_indices,
	const float* proj,
	const float* w2c,
	const float* t_A_inv,
	const bool* visibility_filter,
	const float* dL_dt_A_inv,
	const float* dL_ddepths,
	float* dL_dvertices,
	float* dL_dv_2Ds
);

#endif