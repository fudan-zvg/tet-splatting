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

__global__ void 
GetTetGradientsForwardCUDAKernel(
	const int P,
    const int4* tet_indices,
	const float* sdfs,
	const glm::vec3* vertices,
	glm::vec3* t_gradients,
	glm::mat4* A_invs
){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	const int4 indice = tet_indices[idx];
    glm::vec4 sdf = {sdfs[indice.x], sdfs[indice.y], sdfs[indice.z], sdfs[indice.w]};
    glm::vec3 vertice[4] = {vertices[indice.x], vertices[indice.y], vertices[indice.z], vertices[indice.w]};
	glm::mat4 A = glm::mat4(
		vertice[0].x, vertice[1].x, vertice[2].x, vertice[3].x,
		vertice[0].y, vertice[1].y, vertice[2].y, vertice[3].y,
		vertice[0].z, vertice[1].z, vertice[2].z, vertice[3].z,
		1, 1, 1, 1);	
	glm::mat4 A_inv = glm::inverse(A);
	A_invs[idx] = A_inv;
	glm::vec3 grad(A_inv * sdf);
	t_gradients[idx] = grad;
}

__global__ void 
GetTetGradientsBackwardCUDAKernel(
	const int P,
    const int4* tet_indices,
	const float* sdfs,
	const glm::vec3* vertices,
	const glm::mat4* A_invs,
	const glm::vec3* dL_dt_gradients,
	float* dL_dsdfs,
	float* dL_dvertices
){
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	const int4 indice = tet_indices[idx];
    glm::vec4 sdf = {sdfs[indice.x], sdfs[indice.y], sdfs[indice.z], sdfs[indice.w]};
    glm::vec3 vertice[4] = {vertices[indice.x], vertices[indice.y], vertices[indice.z], vertices[indice.w]};
	glm::mat4 A = glm::mat4(
		vertice[0].x, vertice[1].x, vertice[2].x, vertice[3].x,
		vertice[0].y, vertice[1].y, vertice[2].y, vertice[3].y,
		vertice[0].z, vertice[1].z, vertice[2].z, vertice[3].z,
		1, 1, 1, 1);	
	glm::mat4 A_inv = glm::inverse(A);
	// glm::mat4 A_inv = A_invs[idx];
	glm::mat4 A_inv_t = glm::transpose(A_inv);
	glm::vec4 dL_dt_gradient_homo(dL_dt_gradients[idx], 0);
	glm::vec4 dL_dsdf = glm::transpose(A_inv) * dL_dt_gradient_homo;
	glm::mat4 dL_dA_inv = glm::outerProduct(dL_dt_gradient_homo, sdf);
	glm::mat4 dL_dA = -A_inv_t * dL_dA_inv * A_inv_t;
	glm::mat4 A_inv_t2 = A_inv_t * dL_dA_inv;

	atomicAdd(&(dL_dsdfs[indice.x]), dL_dsdf.x);
	atomicAdd(&(dL_dsdfs[indice.y]), dL_dsdf.y);
	atomicAdd(&(dL_dsdfs[indice.z]), dL_dsdf.z);
	atomicAdd(&(dL_dsdfs[indice.w]), dL_dsdf.w);

	float* dL_dvertice_x = dL_dvertices + 3 * indice.x;
	float* dL_dvertice_y = dL_dvertices + 3 * indice.y;
	float* dL_dvertice_z = dL_dvertices + 3 * indice.z;
	atomicAdd(&(dL_dvertice_x[0]), dL_dA[0][0]);
	atomicAdd(&(dL_dvertice_y[0]), dL_dA[0][1]);
	atomicAdd(&(dL_dvertice_z[0]), dL_dA[0][2]);
	atomicAdd(&(dL_dvertice_x[1]), dL_dA[1][0]);
	atomicAdd(&(dL_dvertice_y[1]), dL_dA[1][1]);
	atomicAdd(&(dL_dvertice_z[1]), dL_dA[1][2]);
	atomicAdd(&(dL_dvertice_x[2]), dL_dA[2][0]);
	atomicAdd(&(dL_dvertice_y[2]), dL_dA[2][1]);
	atomicAdd(&(dL_dvertice_z[2]), dL_dA[2][2]);
}

void GetTetGradientsForwardCUDA(
	const int P,
	const int* tet_indices,
	const float* sdfs,
	const float* vertices,
	float* t_gradients,
	float* A_invs
){
    GetTetGradientsForwardCUDAKernel <<<(P + 255) / 256, 256>>> (
		P,
		(const int4*)tet_indices,
		sdfs,
		(const glm::vec3*)vertices,
		(glm::vec3*)t_gradients,
		(glm::mat4*)A_invs
		);
}

void GetTetGradientsBackwardCUDA(
	const int P,
	const int* tet_indices,
	const float* sdfs,
	const float* vertices,
	const float* A_invs,
	const float* dL_dt_gradients,
	float* dL_dsdfs,
	float* dL_dvertices
){
	GetTetGradientsBackwardCUDAKernel <<<(P + 255) / 256, 256>>> (
		P,
		(const int4*)tet_indices,
		sdfs,
		(const glm::vec3*)vertices,
		(const glm::mat4*)A_invs,
		(const glm::vec3*)dL_dt_gradients,
		dL_dsdfs,
		dL_dvertices
		);
}