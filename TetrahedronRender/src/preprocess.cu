#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include "config.h"
#include "auxiliary.h"

__global__ void 
preprcess_vertices_kernel(
    const int V, const int H, const int W,
	const float* vertices,
    const float* proj,
	const float* w2c,
    float* depths,
    float2* v_2Ds
){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= V)
		return;
    float3 vertice = {vertices[idx * 3], vertices[idx * 3 + 1], vertices[idx * 3 + 2]};
    depths[idx] = -(w2c[8] * vertice.x + w2c[9] * vertice.y + w2c[10] * vertice.z + w2c[11]);

    float v_w = 1 / (proj[12] * vertice.x + proj[13] * vertice.y + proj[14] * vertice.z + proj[15] + 0.0000001f);
    float2 v_proj_homo = {
		(proj[0] * vertice.x + proj[1] * vertice.y + proj[2] * vertice.z + proj[3]) * v_w,
		(proj[4] * vertice.x + proj[5] * vertice.y + proj[6] * vertice.z + proj[7]) * v_w
	};
    v_2Ds[idx] = {
        ((v_proj_homo.x + 1) * W - 1) * 0.5f,
        ((v_proj_homo.y + 1) * H - 1) * 0.5f
    };
}

__global__ void 
preprcess_vertices_backward_kernel(
    const int V, const int H, const int W,
	const float* vertices,
    const float* proj,
	const float* w2c,
    const float* dL_ddepths,
    float3* dL_dvertices,
    const float2* dL_dv_2Ds
){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= V)
		return;
    float3 vertice = {vertices[idx * 3], vertices[idx * 3 + 1], vertices[idx * 3 + 2]};
    float v_w = 1 / (proj[12] * vertice.x + proj[13] * vertice.y + proj[14] * vertice.z + proj[15] + 0.0000001f);
    float2 v_proj_homo = {
		(proj[0] * vertice.x + proj[1] * vertice.y + proj[2] * vertice.z + proj[3]),
		(proj[4] * vertice.x + proj[5] * vertice.y + proj[6] * vertice.z + proj[7])
	};

    float2 dL_dv_2D = dL_dv_2Ds[idx];
    float2 dL_dv_proj_homo2 = {
        0.5f * W * dL_dv_2D.x,
        0.5f * H * dL_dv_2D.y
    };

    float2 dL_dv_proj_homo = {
        dL_dv_proj_homo2.x * v_w,
        dL_dv_proj_homo2.y * v_w
    };

    float dL_dv_w = dL_dv_proj_homo2.x * v_proj_homo.x + dL_dv_proj_homo2.y * v_proj_homo.y;
    float dL_dv_w_inv = dL_dv_w * (-v_w*v_w);

    float dL_ddepth = dL_ddepths[idx];
    float3 dL_dvertice = {
        (dL_dv_proj_homo.x * proj[0] + dL_dv_proj_homo.y * proj[4]) + dL_dv_w_inv * proj[12] - dL_ddepth * w2c[8],
        (dL_dv_proj_homo.x * proj[1] + dL_dv_proj_homo.y * proj[5]) + dL_dv_w_inv * proj[13] - dL_ddepth * w2c[9],
        (dL_dv_proj_homo.x * proj[2] + dL_dv_proj_homo.y * proj[6]) + dL_dv_w_inv * proj[14] - dL_ddepth * w2c[10]
    };
    dL_dvertices[idx] = dL_dvertice;
}

__global__ void 
preprcess_tiles_kernel(
    const int P, const int H, const int W, const int2 grid,
	const int4* tet_indices,
    const float* depths,
	const float2* v_2Ds,
    float* depths_to_sort,
    glm::mat3* t_A_inv,
    int4* t_proj_2D_min_max,
    int4* rect,
    int* tiles_touched,
    bool* visibility_filter
){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    int4 tet_indice = tet_indices[idx];
    float depth[4] = {depths[tet_indice.x], depths[tet_indice.y], depths[tet_indice.z], depths[tet_indice.w]};
    depths_to_sort[idx] = (depth[0] + depth[1] + depth[2] + depth[3]) / 4;
    const bool visible = fminf(depth[0], fminf(depth[1], fminf(depth[2], depth[3]))) > 0.01;
    visibility_filter[idx] = true;

    const float2 t_2D[4] = {v_2Ds[tet_indice.x], v_2Ds[tet_indice.y], v_2Ds[tet_indice.z], v_2Ds[tet_indice.w]};
    const int4 proj_2D_min_max = {
        (int)fminf(t_2D[0].x, fminf(t_2D[1].x, fminf(t_2D[2].x, t_2D[3].x))),
        (int)fminf(t_2D[0].y, fminf(t_2D[1].y, fminf(t_2D[2].y, t_2D[3].y))),
        (int)ceilf(fmaxf(t_2D[0].x, fmaxf(t_2D[1].x, fmaxf(t_2D[2].x, t_2D[3].x)))),
        (int)ceilf(fmaxf(t_2D[0].y, fmaxf(t_2D[1].y, fmaxf(t_2D[2].y, t_2D[3].y)))),
    };

    t_proj_2D_min_max[idx] = proj_2D_min_max;
    int4 rect_this = {
        min(grid.x, max(0, proj_2D_min_max.x / BLOCK_X)),
        min(grid.y, max(0, proj_2D_min_max.y / BLOCK_X)),
        min(grid.x, max(0, (proj_2D_min_max.z + BLOCK_X - 1) / BLOCK_X)),
        min(grid.y, max(0, (proj_2D_min_max.w + BLOCK_X - 1) / BLOCK_X))
    };
    int tile = (rect_this.z - rect_this.x) * (rect_this.w - rect_this.y);
    glm::mat3* t_A_inv_this = t_A_inv + 4 * idx;

    glm::mat3 t_A_this_0 = glm::mat3(
        t_2D[0].x, t_2D[1].x, t_2D[2].x,
        t_2D[0].y, t_2D[1].y, t_2D[2].y,
        1, 1, 1
    );
    float det_0 = glm::determinant(t_A_this_0);
    if (det_0 == 0){
        visibility_filter[idx] = false;
        return;
    }
    t_A_inv_this[0] = glm::mat3(
        t_2D[1].y-t_2D[2].y, t_2D[2].x-t_2D[1].x, t_2D[1].x*t_2D[2].y - t_2D[2].x*t_2D[1].y,
        t_2D[2].y - t_2D[0].y, t_2D[0].x - t_2D[2].x, t_2D[2].x*t_2D[0].y - t_2D[0].x*t_2D[2].y,
        t_2D[0].y - t_2D[1].y, t_2D[1].x - t_2D[0].x, t_2D[0].x*t_2D[1].y - t_2D[1].x*t_2D[0].y
    ) / det_0;

    glm::mat3 t_A_this_1 = glm::mat3(
        t_2D[0].x, t_2D[1].x, t_2D[3].x,
        t_2D[0].y, t_2D[1].y, t_2D[3].y,
        1, 1, 1
    );
    float det_1 = glm::determinant(t_A_this_1);
    if (det_1 == 0){
        visibility_filter[idx] = false;
        return;
    }
    t_A_inv_this[1] = glm::mat3(
        t_2D[1].y-t_2D[3].y, t_2D[3].x-t_2D[1].x, t_2D[1].x*t_2D[3].y - t_2D[3].x*t_2D[1].y,
        t_2D[3].y - t_2D[0].y, t_2D[0].x - t_2D[3].x, t_2D[3].x*t_2D[0].y - t_2D[0].x*t_2D[3].y,
        t_2D[0].y - t_2D[1].y, t_2D[1].x - t_2D[0].x, t_2D[0].x*t_2D[1].y - t_2D[1].x*t_2D[0].y
    ) / det_1;

    glm::mat3 t_A_this_2 = glm::mat3(
        t_2D[0].x, t_2D[2].x, t_2D[3].x,
        t_2D[0].y, t_2D[2].y, t_2D[3].y,
        1, 1, 1
    );
    float det_2 = glm::determinant(t_A_this_2);
    if (det_2 == 0){
        visibility_filter[idx] = false;
        return;
    }
    t_A_inv_this[2] = glm::mat3(
        t_2D[2].y-t_2D[3].y, t_2D[3].x-t_2D[2].x, t_2D[2].x*t_2D[3].y - t_2D[3].x*t_2D[2].y,
        t_2D[3].y - t_2D[0].y, t_2D[0].x - t_2D[3].x, t_2D[3].x*t_2D[0].y - t_2D[0].x*t_2D[3].y,
        t_2D[0].y - t_2D[2].y, t_2D[2].x - t_2D[0].x, t_2D[0].x*t_2D[2].y - t_2D[2].x*t_2D[0].y
    ) / det_2;

    glm::mat3 t_A_this_3 = glm::mat3(
        t_2D[1].x, t_2D[2].x, t_2D[3].x,
        t_2D[1].y, t_2D[2].y, t_2D[3].y,
        1, 1, 1
    );
    float det_3 = glm::determinant(t_A_this_3);
    if (det_3 == 0){
        visibility_filter[idx] = false;
        return;
    }
    t_A_inv_this[3] = glm::mat3(
        t_2D[2].y-t_2D[3].y, t_2D[3].x-t_2D[2].x, t_2D[2].x*t_2D[3].y - t_2D[3].x*t_2D[2].y,
        t_2D[3].y - t_2D[1].y, t_2D[1].x - t_2D[3].x, t_2D[3].x*t_2D[1].y - t_2D[1].x*t_2D[3].y,
        t_2D[1].y - t_2D[2].y, t_2D[2].x - t_2D[1].x, t_2D[1].x*t_2D[2].y - t_2D[2].x*t_2D[1].y
    ) / det_3;

    tiles_touched[idx] = tile;
    rect[idx] = rect_this;
}


__global__ void 
preprcess_tiles_backward_kernel(
    const int P, const int H, const int W,
	const int4* tet_indices,
    const glm::mat3* t_A_inv,
    const glm::mat3* dL_dt_A_inv,
    const bool* visibility_filter,
    float2* dL_dv_2Ds
){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !visibility_filter[idx])
		return;

    int4 tet_indice = tet_indices[idx];
    float2 dL_dt_2D[4] = {0.0f};
    const glm::mat3* t_A_inv_this = t_A_inv + 4 * idx;
    const glm::mat3* dL_dt_A_inv_this = dL_dt_A_inv + 4 * idx;

    const int3 indices[4] = {
        {0, 1, 2},
        {0, 1, 3},
        {0, 2, 3},
        {1, 2, 3}
    };
    for (int i=0;i<4;i++){
        glm::mat3 A_inv = t_A_inv_this[i];
        glm::mat3 A_inv_t = glm::transpose(A_inv);
        glm::mat3 dL_dA_inv = dL_dt_A_inv_this[i];
        glm::mat3 dL_dA = -A_inv_t * dL_dA_inv * A_inv_t;
        const int3 indice = indices[i];
        dL_dt_2D[indice.x].x += dL_dA[0][0];
        dL_dt_2D[indice.y].x += dL_dA[0][1];
        dL_dt_2D[indice.z].x += dL_dA[0][2];
        dL_dt_2D[indice.x].y += dL_dA[1][0];
        dL_dt_2D[indice.y].y += dL_dA[1][1];
        dL_dt_2D[indice.z].y += dL_dA[1][2];
    }
    atomicAdd(&(dL_dv_2Ds[tet_indice.x].x), dL_dt_2D[0].x);
    atomicAdd(&(dL_dv_2Ds[tet_indice.x].y), dL_dt_2D[0].y);
    atomicAdd(&(dL_dv_2Ds[tet_indice.y].x), dL_dt_2D[1].x);
    atomicAdd(&(dL_dv_2Ds[tet_indice.y].y), dL_dt_2D[1].y);
    atomicAdd(&(dL_dv_2Ds[tet_indice.z].x), dL_dt_2D[2].x);
    atomicAdd(&(dL_dv_2Ds[tet_indice.z].y), dL_dt_2D[2].y);
    atomicAdd(&(dL_dv_2Ds[tet_indice.w].x), dL_dt_2D[3].x);
    atomicAdd(&(dL_dv_2Ds[tet_indice.w].y), dL_dt_2D[3].y);
}

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
){
    preprcess_vertices_kernel <<<(V + 255) / 256, 256>>> (
        V, H, W,
        vertices,
        proj,
        w2c,
        depths,
        (float2*)v_2Ds
    );

    int2 grid = {(W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y};
    preprcess_tiles_kernel <<<(P + 255) / 256, 256>>> (
        P, H, W, grid,
        (const int4*)tet_indices,
        depths,
        (const float2*)v_2Ds,
        depths_to_sort,
        (glm::mat3*)t_A_inv,
        (int4*)t_proj_2D_min_max,
        (int4*)rect,
        tiles_touched,
        visibility_filter
    );
}

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
){
    preprcess_tiles_backward_kernel <<<(P + 255) / 256, 256>>> (
        P, H, W,
        (const int4*)tet_indices,
        (const glm::mat3*)t_A_inv,
        (const glm::mat3*)dL_dt_A_inv,
        visibility_filter,
        (float2*)dL_dv_2Ds
    );

    preprcess_vertices_backward_kernel <<<(V + 255) / 256, 256>>> (
        V, H, W,
        vertices,
        proj,
        w2c,
        dL_ddepths,
        (float3*)dL_dvertices,
        (const float2*)dL_dv_2Ds
    );
}