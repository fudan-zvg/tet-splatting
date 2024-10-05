#ifndef CUDA_VISIBLE_TET_H_INCLUDED
#define CUDA_VISIBLE_TET_H_INCLUDED

void GetVisibleTetCUDA(
	const int P,
	const float* sdfs_var,
	const int* tet_indices,
	bool* t_mask,
	float alpha_threshold);

#endif
