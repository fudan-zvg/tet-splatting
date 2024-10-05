#ifndef CUDA_TET_GRADIENTS_H_INCLUDED
#define CUDA_TET_GRADIENTS_H_INCLUDED

void GetTetGradientsForwardCUDA(
	const int P,
	const int* tet_indices,
	const float* sdfs,
	const float* vertices,
	float* t_gradients,
	float* A_invs);

void GetTetGradientsBackwardCUDA(
	const int P,
	const int* tet_indices,
	const float* sdfs,
	const float* vertices,
	const float* A_invs,
	const float* dL_dt_gradients,
	float* dL_dsdfs,
	float* dL_dvertices);

#endif
