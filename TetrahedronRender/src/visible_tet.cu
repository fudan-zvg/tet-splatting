#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void 
GetVisibleTetCUDAKernel(
	const int P,
    const float* sdfs_var,
    const int4* tet_indices,
	bool* t_mask,
	float alpha_threshold
){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	const int4 tet_idx = tet_indices[idx];
    float sdfs[4] = {sdfs_var[tet_idx.x], sdfs_var[tet_idx.y], sdfs_var[tet_idx.z], sdfs_var[tet_idx.w]};
    float sdf_var_max = fmaxf(fmaxf(sdfs[0], sdfs[1]), fmaxf(sdfs[2], sdfs[3]));
    float sdf_var_min = fminf(fminf(sdfs[0], sdfs[1]), fminf(sdfs[2], sdfs[3]));
    t_mask[idx] = 1 - fmaxf(1.0f + __expf(-sdf_var_max),0.00001f) / fmaxf(1.0f + __expf(-sdf_var_min),0.00001f) > alpha_threshold;
}


void GetVisibleTetCUDA(
	const int P,
	const float* sdfs_var,
	const int* tet_indices,
	bool* t_mask,
	float alpha_threshold
){
    GetVisibleTetCUDAKernel <<<(P + 255) / 256, 256>>> (
		P,
		sdfs_var,
		(const int4*)tet_indices,
		t_mask,
		alpha_threshold);
}