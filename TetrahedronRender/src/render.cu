#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include "config.h"
#include "auxiliary.h"
#define RESORT


BinningState BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}
// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	const int P,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* tet_keys_unsorted,
	uint32_t* tet_values_unsorted,
	const bool* visibility_filter,
	const int32_t* rect,
	const int32_t* tiles_touched,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !visibility_filter[idx])
		return;

	// Find this Gaussian's offset in buffer for writing keys/values.
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	int2 rect_min = {rect[idx*4], rect[idx*4+1]}, rect_max = {rect[idx*4+2], rect[idx*4+3]};
	// For each tile that the bounding rect overlaps, emit a 
	// key/value pair. The key is |  tile ID  |      depth      |,
	// and the value is the ID of the Gaussian. Sorting the values 
	// with this key yields Gaussian IDs in a list, such that they
	// are first sorted by tile and then by depth. 
	for (int y = rect_min.y; y < rect_max.y; y++)
	{
		for (int x = rect_min.x; x < rect_max.x; x++)
		{
			uint64_t key = y * grid.x + x;
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);
			tet_keys_unsorted[off] = key;
			tet_values_unsorted[off] = idx;
			off++;
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, int2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

template <uint32_t MAX_C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
RenderForwardCUDAKernel(
	const int S, 
	const int2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ t_A_inv,
	const float* __restrict__ t_features,
	const float* __restrict__ t_depths,
	const float* __restrict__ t_sdfs_var,
	const int* __restrict__ t_proj_2D_min_max,
	int* __restrict__ contrib,
	float* __restrict__ out_alpha,
	float* __restrict__ out_feature,
	float* __restrict__ error_map,
	float alpha_threshold
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	int2 pix_int = { (int)pix.x, (int)pix.y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W&& pix.y < H;
	bool done = !inside;

	int2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_u_coef[4*BLOCK_SIZE];
	__shared__ float3 collected_v_coef[4*BLOCK_SIZE];
	__shared__ float collected_depth[4*BLOCK_SIZE];
	__shared__ int4 collected_t_proj_2D_min_max[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	float O = 0.0f, F[MAX_C] = {0.0f};
	float last_z = -1.;
	float error=0.0f;
#ifdef RESORT
	int sort_global_ids[WINDOW_SIZE];
	int6 sort_indices[WINDOW_SIZE];
	float6 sort_uvzs[WINDOW_SIZE];
	int sort_num = 0;
	for (int i = 0; i < WINDOW_SIZE; ++i)
	{
		sort_global_ids[i] = -1;
		sort_indices[i] = {0, 0, 0, 0, 0, 0};
		sort_uvzs[i] = {0, 0, FLT_MAX, 0, 0, 0};
	}

	auto blend_one = [&]() {
		if (sort_num == 0)
			return;
		--sort_num;
		int global_id = sort_global_ids[0];

		const float* sdf_ptr = t_sdfs_var + global_id*4;
		
		int6 indices = sort_indices[0];
		int3 indices_0 = {indices.x, indices.y, indices.z};
		int3 indices_1 = {indices.w, indices.u, indices.v};

		float6 uvzs = sort_uvzs[0];
		float3 uvz_0 = {uvzs.x, uvzs.y, uvzs.z};
		float3 uvz_1 = {uvzs.w, uvzs.u, uvzs.v};
		const float sdf_prev = sdf_ptr[indices_0.x] * (1-uvz_0.x-uvz_0.y) + sdf_ptr[indices_0.y] * uvz_0.x + sdf_ptr[indices_0.z] * uvz_0.y;
		const float sdf_next = sdf_ptr[indices_1.x] * (1-uvz_1.x-uvz_1.y) + sdf_ptr[indices_1.y] * uvz_1.x + sdf_ptr[indices_1.z] * uvz_1.y;

		float alpha = max(0.0f, 1 - max(sigmoid(sdf_next),0.00001f) / max(sigmoid(sdf_prev),0.00001f));
		if (alpha < alpha_threshold)
			return;
		contributor++;
		float weight = alpha * T;
		T *= 1 - alpha;
		
		O += weight;

		const float* t_features_ptr = t_features + global_id*S;
		for (int ch = 0; ch < S; ch++){
			F[ch] += weight * t_features_ptr[ch];
		}

		error += max(last_z - uvz_0.z, 0.0f);
		last_z = uvz_0.z;

		if (T < 0.0001f){ 
			done = true;
		}

		for (int i = 1; i < WINDOW_SIZE; ++i)
		{
			sort_global_ids[i - 1] = sort_global_ids[i];
			sort_indices[i - 1] = sort_indices[i];
			sort_uvzs[i - 1] = sort_uvzs[i];
		}
		sort_uvzs[WINDOW_SIZE - 1].z = FLT_MAX;
	};
#endif

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			const float* A_inv_this = t_A_inv + 36 * coll_id;
			for (int k=0;k<4;k++){
				const float* A_inv_cur = A_inv_this + 9 * k;
				collected_u_coef[k*BLOCK_SIZE + block.thread_rank()] = {A_inv_cur[3], A_inv_cur[4], A_inv_cur[5]};
				collected_v_coef[k*BLOCK_SIZE + block.thread_rank()] = {A_inv_cur[6], A_inv_cur[7], A_inv_cur[8]};
				collected_depth[k*BLOCK_SIZE + block.thread_rank()] = t_depths[4*coll_id+k];
			}
			collected_id[block.thread_rank()] = coll_id;
			collected_t_proj_2D_min_max[block.thread_rank()] = {t_proj_2D_min_max[coll_id*4],t_proj_2D_min_max[coll_id*4+1],
																t_proj_2D_min_max[coll_id*4+2],t_proj_2D_min_max[coll_id*4+3]};
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
#ifdef RESORT
			if (sort_num == WINDOW_SIZE) {
				blend_one();
			}
#endif
			if (done == true)
				break;

			int global_id = collected_id[j];
			int4 t_proj_2D = collected_t_proj_2D_min_max[j];
			if (pix_int.x < t_proj_2D.x || pix_int.y < t_proj_2D.y || pix_int.x > t_proj_2D.z || pix_int.y > t_proj_2D.w){
				continue;
			}

			float z_0 = collected_depth[j];
			float z_1 = collected_depth[BLOCK_SIZE+j];
			float z_2 = collected_depth[2*BLOCK_SIZE+j];
			float z_3 = collected_depth[3*BLOCK_SIZE+j];

			int count = 0;
			float3 uvzs[3];
			int3 indices[3];

			bool in[4]={0};
			int count_in = 0;
			float3 u_coef_0 = collected_u_coef[j];
			float3 v_coef_0 = collected_v_coef[j];
			float3 u_coef_1 = collected_u_coef[BLOCK_SIZE+j];
			float3 v_coef_1 = collected_v_coef[BLOCK_SIZE+j];
			float3 u_coef_2 = collected_u_coef[2*BLOCK_SIZE+j];
			float3 v_coef_2 = collected_v_coef[2*BLOCK_SIZE+j];
			float3 u_coef_3 = collected_u_coef[3*BLOCK_SIZE+j];
			float3 v_coef_3 = collected_v_coef[3*BLOCK_SIZE+j];

			// sometimes there is one or three hits due to numerical precision
			float eps = 0.00001;
			float u_0 = pixf.x * u_coef_0.x + pixf.y * u_coef_0.y + u_coef_0.z;
			float v_0 = pixf.x * v_coef_0.x + pixf.y * v_coef_0.y + v_coef_0.z;
			float err_0 = max(max(max(-u_0,0.0f),max(-v_0,0.0f)),max(u_0+v_0-1,0.0f)); 
			if (err_0<=eps){ in[0]=true; count_in++;}

			float u_1 = pixf.x * u_coef_1.x + pixf.y * u_coef_1.y + u_coef_1.z;
			float v_1 = pixf.x * v_coef_1.x + pixf.y * v_coef_1.y + v_coef_1.z;
			float err_1 = max(max(max(-u_1,0.0f),max(-v_1,0.0f)),max(u_1+v_1-1,0.0f)); 
			if (err_1<=eps){ in[1]=true; count_in++;}

			float u_2 = pixf.x * u_coef_2.x + pixf.y * u_coef_2.y + u_coef_2.z;
			float v_2 = pixf.x * v_coef_2.x + pixf.y * v_coef_2.y + v_coef_2.z;
			float err_2 = max(max(max(-u_2,0.0f),max(-v_2,0.0f)),max(u_2+v_2-1,0.0f)); 
			if (err_2<=eps){ in[2]=true; count_in++;}
			
			float u_3 = pixf.x * u_coef_3.x + pixf.y * u_coef_3.y + u_coef_3.z;
			float v_3 = pixf.x * v_coef_3.x + pixf.y * v_coef_3.y + v_coef_3.z;
			float err_3 = max(max(max(-u_3,0.0f),max(-v_3,0.0f)),max(u_3+v_3-1,0.0f)); 
			if (err_3<=eps){ in[3]=true; count_in++;}

			// if ((count_in!=2) && (count_in!=0)) printf("count_in:%d\n",count_in);
			if (count_in<2) continue;

			if (in[0]){
				float3 uvz = projection_correct(u_0, v_0, z_0, z_1, z_2);
				uvzs[count] = uvz;
				indices[count] = {0, 1, 2};
				count++;
			}

			if (in[1]){
				float3 uvz = projection_correct(u_1, v_1, z_0, z_1, z_3);
				uvzs[count] = uvz;
				indices[count] = {0, 1, 3};
				count++;
			}

			if (in[2]){
				float3 uvz = projection_correct(u_2, v_2, z_0, z_2, z_3);
				bool cond = ((count == 2) && (abs(uvzs[0].z-uvzs[1].z)<abs(uvzs[0].z-uvz.z)));
				if ((count < 2) || cond){ 
					if (cond) count--;
					uvzs[count] = uvz;
					indices[count] = {0, 2, 3};
					count++;
				}
			}

			if (in[3]){
				float3 uvz = projection_correct(u_3, v_3, z_1, z_2, z_3);
				bool cond = ((count == 2) && (abs(uvzs[0].z-uvzs[1].z)<abs(uvzs[0].z-uvz.z)));
				if ((count < 2) || cond){ 
					if (cond) count--;
					uvzs[count] = uvz;
					indices[count] = {1, 2, 3};
					count++;
				}
			}

			if (uvzs[0].z>uvzs[1].z) {
				swap(indices[0], indices[1]);
				swap(uvzs[0], uvzs[1]);
			}

			const float* sdf_ptr = t_sdfs_var + global_id*4;
			
			int3 indices_0 = indices[0];
			int3 indices_1 = indices[1];
			float3 uvz_0 = uvzs[0];
			float3 uvz_1 = uvzs[1];
			const float sdf_prev = sdf_ptr[indices_0.x] * (1-uvz_0.x-uvz_0.y) + sdf_ptr[indices_0.y] * uvz_0.x + sdf_ptr[indices_0.z] * uvz_0.y;
			const float sdf_next = sdf_ptr[indices_1.x] * (1-uvz_1.x-uvz_1.y) + sdf_ptr[indices_1.y] * uvz_1.x + sdf_ptr[indices_1.z] * uvz_1.y;

			float alpha = max(0.0f, 1 - max(sigmoid(sdf_next),0.00001f) / max(sigmoid(sdf_prev),0.00001f));
			if (alpha < alpha_threshold)
				continue;

#ifdef RESORT
			float6 _uvzs(uvzs[0].x, uvzs[0].y, uvzs[0].z, uvzs[1].x, uvzs[1].y, uvzs[1].z);
			int6 _indices(indices[0].x, indices[0].y, indices[0].z, indices[1].x, indices[1].y, indices[1].z);
			
			#pragma unroll
			for (int s = 0; s < WINDOW_SIZE; ++s) 
			{
				if (_uvzs.z < sort_uvzs[s].z) 
				{
					swap(global_id , sort_global_ids[s]);
					swap(_indices, sort_indices[s]);
					swap(_uvzs, sort_uvzs[s]);
				}
			}
			++sort_num;
#else
			contributor++;
			float weight = alpha * T;
			T *= 1 - alpha;
			
			O += weight;

			const float* t_features_ptr = t_features + global_id*S;
			for (int ch = 0; ch < S; ch++){
				F[ch] += weight * t_features_ptr[ch];
			}

			error += max(last_z - uvz_0.z, 0.0f);
			last_z = uvz_0.z;

			if (T < 0.0001f){ 
				done = true;
			}
#endif
		}
	}

#ifdef RESORT
	if (!done) {
		while (sort_num > 0)
			blend_one();
	}
#endif
	if (inside)
	{
		error_map[pix_id] = error;
		contrib[pix_id] = contributor;
		out_alpha[pix_id] = O;
		for (int ch = 0; ch < S; ch++)
			out_feature[ch * H * W + pix_id] = F[ch];
	}
}


template <uint32_t MAX_C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
RenderBackwardCUDAKernel(
	const int S, 
	const int2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ t_A_inv,
	const float* __restrict__ t_features,
	const float* __restrict__ t_depths,
	const float* __restrict__ t_sdfs_var,
	const int* __restrict__ t_proj_2D_min_max,
	const float* __restrict__ out_alpha,
	const float* __restrict__ out_feature,
	const int* __restrict__ contrib,
	const float* __restrict__ dL_dout_alpha,
	const float* __restrict__ dL_dout_feature,
	float* __restrict__ dL_dt_A_inv,
	float* __restrict__ dL_dt_features,
	float* __restrict__ dL_dt_depths,
	float* __restrict__ dL_dt_sdfs_var,
	float alpha_threshold
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const int2 pix_int = { (int)pix.x, (int)pix.y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	bool done = !inside;

	const int2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_u_coef[4*BLOCK_SIZE];
	__shared__ float3 collected_v_coef[4*BLOCK_SIZE];
	__shared__ float collected_depth[4*BLOCK_SIZE];
	__shared__ int4 collected_t_proj_2D_min_max[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	float O, o=0.0f, F[MAX_C], f[MAX_C]={0.0f};
	float dL_dO, dL_dF[MAX_C];
	if (inside){
		O = out_alpha[pix_id];
		dL_dO = dL_dout_alpha[pix_id];
		for (int ch = 0; ch < S; ch++){
			F[ch] = out_feature[ch * H * W + pix_id];
			dL_dF[ch] = dL_dout_feature[ch * H * W + pix_id];
		}
	}

#ifdef RESORT
	int sort_global_ids[WINDOW_SIZE];
	int6 sort_indices[WINDOW_SIZE];
	float6 sort_uvzs[WINDOW_SIZE];
	float4 sort_uvs_before[WINDOW_SIZE];
	float6 sort_zs_before[WINDOW_SIZE];
	int2 sort_ids[WINDOW_SIZE];
	int sort_num = 0;
	for (int i = 0; i < WINDOW_SIZE; ++i)
	{
		sort_global_ids[i] = -1;
		sort_indices[i] = {0, 0, 0, 0, 0, 0};
		sort_uvzs[i] = {0, 0, FLT_MAX, 0, 0, 0};
		sort_uvs_before[i] = {0, 0, 0, 0};
		sort_zs_before[i] = {0, 0, 0, 0, 0, 0};
		sort_ids[i] = {0, 0};
	}

	auto blend_one = [&]() {
		if (sort_num == 0)
			return;
		contributor++;
		--sort_num;
		int global_id = sort_global_ids[0];

		const float* sdf_ptr = t_sdfs_var + global_id*4;
			
		int6 indices = sort_indices[0];
		int3 indices_0 = {indices.x, indices.y, indices.z};
		int3 indices_1 = {indices.w, indices.u, indices.v};

		float6 uvzs = sort_uvzs[0];
		float3 uvz_0 = {uvzs.x, uvzs.y, uvzs.z}, dL_duvz_0={0,0,0};
		float3 uvz_1 = {uvzs.w, uvzs.u, uvzs.v}, dL_duvz_1={0,0,0};
		const float sdf_prev = sdf_ptr[indices_0.x] * (1-uvz_0.x-uvz_0.y) + sdf_ptr[indices_0.y] * uvz_0.x + sdf_ptr[indices_0.z] * uvz_0.y;
		const float sdf_next = sdf_ptr[indices_1.x] * (1-uvz_1.x-uvz_1.y) + sdf_ptr[indices_1.y] * uvz_1.x + sdf_ptr[indices_1.z] * uvz_1.y;

		const float sdf_prev_sig = sigmoid(sdf_prev);
		const float sdf_next_sig = sigmoid(sdf_next);
		float alpha_before = 1 - max(sdf_next_sig,0.00001f) / max(sdf_prev_sig,0.00001f);
		float alpha = max(0.0f, alpha_before);

		if (alpha < alpha_threshold)
			return;
		contributor++;
		float weight = alpha * T;
		T *= 1 - alpha;
		
		float dL_dalpha = 0.0f;
		o += weight;
		dL_dalpha += dL_dO * (1 - O);

		const float* t_features_ptr = t_features + global_id*S;
		float* dL_dt_features_ptr = dL_dt_features + global_id*S;
		for (int ch = 0; ch < S; ch++){
			float fea = t_features_ptr[ch];
			f[ch] += weight * fea;
			atomicAdd(&(dL_dt_features_ptr[ch]), weight * dL_dF[ch]);
			dL_dalpha += dL_dF[ch] * (T * fea - (F[ch] - f[ch]));
		}

		dL_dalpha /= max(0.00001f, 1.0f - alpha);

		if (alpha_before >= 0.0f){
			float dL_dsdf_prev_sig = dL_dalpha * (max(sdf_next_sig,0.00001f) / (max(sdf_prev_sig,0.00001f) * max(sdf_prev_sig,0.00001f)));
			float dL_dsdf_next_sig = dL_dalpha * (-1.0f / max(sdf_prev_sig,0.00001f));

			float dL_dsdf_prev = dL_dsdf_prev_sig * sdf_prev_sig * (1 - sdf_prev_sig);
			float dL_dsdf_next = dL_dsdf_next_sig * sdf_next_sig * (1 - sdf_next_sig);

			float* dL_dsdf_ptr = dL_dt_sdfs_var + global_id * 4;
			atomicAdd(&(dL_dsdf_ptr[indices_0.x]), dL_dsdf_prev * (1-uvz_0.x-uvz_0.y));
			atomicAdd(&(dL_dsdf_ptr[indices_0.y]), dL_dsdf_prev * uvz_0.x);
			atomicAdd(&(dL_dsdf_ptr[indices_0.z]), dL_dsdf_prev * uvz_0.y);

			atomicAdd(&(dL_dsdf_ptr[indices_1.x]), dL_dsdf_next * (1-uvz_1.x-uvz_1.y));
			atomicAdd(&(dL_dsdf_ptr[indices_1.y]), dL_dsdf_next * uvz_1.x);
			atomicAdd(&(dL_dsdf_ptr[indices_1.z]), dL_dsdf_next * uvz_1.y);

			dL_duvz_0.x += dL_dsdf_prev * (sdf_ptr[indices_0.y] - sdf_ptr[indices_0.x]);
			dL_duvz_0.y += dL_dsdf_prev * (sdf_ptr[indices_0.z] - sdf_ptr[indices_0.x]);

			dL_duvz_1.x += dL_dsdf_next * (sdf_ptr[indices_1.y] - sdf_ptr[indices_1.x]);
			dL_duvz_1.y += dL_dsdf_next * (sdf_ptr[indices_1.z] - sdf_ptr[indices_1.x]);
		}

		float4 uvs_before = sort_uvs_before[0];
		float6 zs_before = sort_zs_before[0];
		float2 uvs_before_0 = {uvs_before.x, uvs_before.y}, dL_duvs_before_0;
		float3 zs_before_0 = {zs_before.x, zs_before.y, zs_before.z}, dL_dzs_before_0;
		projection_correct_backward(uvs_before_0.x, uvs_before_0.y, zs_before_0.x, zs_before_0.y, zs_before_0.z,
			dL_duvz_0, dL_duvs_before_0.x, dL_duvs_before_0.y, dL_dzs_before_0.x, dL_dzs_before_0.y, dL_dzs_before_0.z);

		int2 ids = sort_ids[0];
		int id_0 = ids.x;
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 3]), dL_duvs_before_0.x * pixf.x);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 4]), dL_duvs_before_0.x * pixf.y);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 5]), dL_duvs_before_0.x);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 6]), dL_duvs_before_0.y * pixf.x);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 7]), dL_duvs_before_0.y * pixf.y);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 8]), dL_duvs_before_0.y);

		atomicAdd(&(dL_dt_depths[4 * global_id + indices_0.x]), dL_dzs_before_0.x);
		atomicAdd(&(dL_dt_depths[4 * global_id + indices_0.y]), dL_dzs_before_0.y);
		atomicAdd(&(dL_dt_depths[4 * global_id + indices_0.z]), dL_dzs_before_0.z);

		float2 uvs_before_1 = {uvs_before.z, uvs_before.w}, dL_duvs_before_1;
		float3 zs_before_1 = {zs_before.w, zs_before.u, zs_before.z}, dL_dzs_before_1;
		projection_correct_backward(uvs_before_1.x, uvs_before_1.y, zs_before_1.x, zs_before_1.y, zs_before_1.z,
			dL_duvz_1, dL_duvs_before_1.x, dL_duvs_before_1.y, dL_dzs_before_1.x, dL_dzs_before_1.y, dL_dzs_before_1.z);
		
		int id_1 = ids.y;
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 3]), dL_duvs_before_1.x * pixf.x);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 4]), dL_duvs_before_1.x * pixf.y);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 5]), dL_duvs_before_1.x);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 6]), dL_duvs_before_1.y * pixf.x);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 7]), dL_duvs_before_1.y * pixf.y);
		atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 8]), dL_duvs_before_1.y);

		atomicAdd(&(dL_dt_depths[4 * global_id + indices_1.x]), dL_dzs_before_1.x);
		atomicAdd(&(dL_dt_depths[4 * global_id + indices_1.y]), dL_dzs_before_1.y);
		atomicAdd(&(dL_dt_depths[4 * global_id + indices_1.z]), dL_dzs_before_1.z);

		if (T < 0.0001f){ 
			done = true;
		}

		for (int i = 1; i < WINDOW_SIZE; ++i)
		{
			sort_global_ids[i - 1] = sort_global_ids[i];
			sort_indices[i - 1] = sort_indices[i];
			sort_uvzs[i - 1] = sort_uvzs[i];
			sort_uvs_before[i - 1] = sort_uvs_before[i];
			sort_zs_before[i - 1] = sort_zs_before[i];
			sort_ids[i - 1] = sort_ids[i];
		}
		sort_uvzs[WINDOW_SIZE - 1].z = FLT_MAX;
	};
#endif

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			const float* A_inv_this = t_A_inv + 36 * coll_id;
			for (int k=0;k<4;k++){
				const float* A_inv_cur = A_inv_this + 9 * k;
				collected_u_coef[k*BLOCK_SIZE + block.thread_rank()] = {A_inv_cur[3], A_inv_cur[4], A_inv_cur[5]};
				collected_v_coef[k*BLOCK_SIZE + block.thread_rank()] = {A_inv_cur[6], A_inv_cur[7], A_inv_cur[8]};
				collected_depth[k*BLOCK_SIZE + block.thread_rank()] = t_depths[4*coll_id+k];
			}
			collected_id[block.thread_rank()] = coll_id;
			collected_t_proj_2D_min_max[block.thread_rank()] = {t_proj_2D_min_max[coll_id*4],t_proj_2D_min_max[coll_id*4+1],
																t_proj_2D_min_max[coll_id*4+2],t_proj_2D_min_max[coll_id*4+3]};
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
#ifdef RESORT
			if (sort_num == WINDOW_SIZE) {
				blend_one();
			}
#endif
			if (done == true)
				break;
				
			int global_id = collected_id[j];
			// int global_id = point_list[range.x + i * BLOCK_SIZE + j];
			int4 t_proj_2D = collected_t_proj_2D_min_max[j];
			// int4 t_proj_2D = {t_proj_2D_min_max[global_id*4],t_proj_2D_min_max[global_id*4+1],
			// 				t_proj_2D_min_max[global_id*4+2],t_proj_2D_min_max[global_id*4+3]};
			if (pix_int.x < t_proj_2D.x || pix_int.y < t_proj_2D.y || pix_int.x > t_proj_2D.z || pix_int.y > t_proj_2D.w){
				continue;
			}

			float z_0 = collected_depth[j];
			float z_1 = collected_depth[BLOCK_SIZE+j];
			float z_2 = collected_depth[2*BLOCK_SIZE+j];
			float z_3 = collected_depth[3*BLOCK_SIZE+j];

			int count = 0;
			float3 uvzs[3];
			float2 uvs_before[3];
			float3 zs_before[3];
			int3 indices[3];
			int ids[3];

			bool in[4]={0};
			int count_in = 0;
			float3 u_coef_0 = collected_u_coef[j];
			float3 v_coef_0 = collected_v_coef[j];
			float3 u_coef_1 = collected_u_coef[BLOCK_SIZE+j];
			float3 v_coef_1 = collected_v_coef[BLOCK_SIZE+j];
			float3 u_coef_2 = collected_u_coef[2*BLOCK_SIZE+j];
			float3 v_coef_2 = collected_v_coef[2*BLOCK_SIZE+j];
			float3 u_coef_3 = collected_u_coef[3*BLOCK_SIZE+j];
			float3 v_coef_3 = collected_v_coef[3*BLOCK_SIZE+j];

			float eps = 0.00001;
			float u_0 = pixf.x * u_coef_0.x + pixf.y * u_coef_0.y + u_coef_0.z;
			float v_0 = pixf.x * v_coef_0.x + pixf.y * v_coef_0.y + v_coef_0.z;
			float err_0 = max(max(max(-u_0,0.0f),max(-v_0,0.0f)),max(u_0+v_0-1,0.0f)); 
			if (err_0<=eps){ in[0]=true; count_in++;}

			float u_1 = pixf.x * u_coef_1.x + pixf.y * u_coef_1.y + u_coef_1.z;
			float v_1 = pixf.x * v_coef_1.x + pixf.y * v_coef_1.y + v_coef_1.z;
			float err_1 = max(max(max(-u_1,0.0f),max(-v_1,0.0f)),max(u_1+v_1-1,0.0f)); 
			if (err_1<=eps){ in[1]=true; count_in++;}

			float u_2 = pixf.x * u_coef_2.x + pixf.y * u_coef_2.y + u_coef_2.z;
			float v_2 = pixf.x * v_coef_2.x + pixf.y * v_coef_2.y + v_coef_2.z;
			float err_2 = max(max(max(-u_2,0.0f),max(-v_2,0.0f)),max(u_2+v_2-1,0.0f)); 
			if (err_2<=eps){ in[2]=true; count_in++;}
			
			float u_3 = pixf.x * u_coef_3.x + pixf.y * u_coef_3.y + u_coef_3.z;
			float v_3 = pixf.x * v_coef_3.x + pixf.y * v_coef_3.y + v_coef_3.z;
			float err_3 = max(max(max(-u_3,0.0f),max(-v_3,0.0f)),max(u_3+v_3-1,0.0f)); 
			if (err_3<=eps){ in[3]=true; count_in++;}

			if (count_in<2) continue;

			if (in[0]){
				float3 uvz = projection_correct(u_0, v_0, z_0, z_1, z_2);
				uvzs[count] = uvz;
				uvs_before[count] = {u_0, v_0};
				zs_before[count] = {z_0, z_1, z_2};
				indices[count] = {0, 1, 2};
				ids[count] = 0;
				count++;
			}

			if (in[1]){
				float3 uvz = projection_correct(u_1, v_1, z_0, z_1, z_3);
				uvzs[count] = uvz;
				uvs_before[count] = {u_1, v_1};
				zs_before[count] = {z_0, z_1, z_3};
				indices[count] = {0, 1, 3};
				ids[count] = 1;
				count++;
			}

			if (in[2]){
				float3 uvz = projection_correct(u_2, v_2, z_0, z_2, z_3);
				bool cond = ((count == 2) && (abs(uvzs[0].z-uvzs[1].z)<abs(uvzs[0].z-uvz.z)));
				if ((count < 2) || cond){ 
					if (cond) count--;
					uvzs[count] = uvz;
					uvs_before[count] = {u_2, v_2};
					zs_before[count] = {z_0, z_2, z_3};
					indices[count] = {0, 2, 3};
					ids[count] = 2;
					count++;
				}
			}

			if (in[3]){
				float3 uvz = projection_correct(u_3, v_3, z_1, z_2, z_3);
				bool cond = ((count == 2) && (abs(uvzs[0].z-uvzs[1].z)<abs(uvzs[0].z-uvz.z)));
				if ((count < 2) || cond){ 
					if (cond) count--;
					uvzs[count] = uvz;
					uvs_before[count] = {u_3, v_3};
					zs_before[count] = {z_1, z_2, z_3};
					indices[count] = {1, 2, 3};
					ids[count] = 3;
					count++;
				}
			}
			
			if (uvzs[0].z>uvzs[1].z) {
				swap(indices[0], indices[1]);
				swap(uvzs[0], uvzs[1]);
				swap(uvs_before[0], uvs_before[1]);
				swap(zs_before[0], zs_before[1]);
				swap(ids[0], ids[1]);
			}
			
			const float* sdf_ptr = t_sdfs_var + global_id*4;
			
			int3 indices_0 = indices[0];
			int3 indices_1 = indices[1];
			float3 uvz_0 = uvzs[0];
			float3 uvz_1 = uvzs[1];
			const float sdf_prev = sdf_ptr[indices_0.x] * (1-uvz_0.x-uvz_0.y) + sdf_ptr[indices_0.y] * uvz_0.x + sdf_ptr[indices_0.z] * uvz_0.y;
			const float sdf_next = sdf_ptr[indices_1.x] * (1-uvz_1.x-uvz_1.y) + sdf_ptr[indices_1.y] * uvz_1.x + sdf_ptr[indices_1.z] * uvz_1.y;

			const float sdf_prev_sig = sigmoid(sdf_prev);
			const float sdf_next_sig = sigmoid(sdf_next);
			float alpha_before = 1 - max(sdf_next_sig,0.00001f) / max(sdf_prev_sig,0.00001f);
			float alpha = max(0.0f, alpha_before);

			if (alpha < alpha_threshold)
				continue;

#ifdef RESORT
			float6 _uvzs(uvzs[0].x, uvzs[0].y, uvzs[0].z, uvzs[1].x, uvzs[1].y, uvzs[1].z);
			int6 _indices(indices[0].x, indices[0].y, indices[0].z, indices[1].x, indices[1].y, indices[1].z);
			float4 _uvs_before = make_float4(uvs_before[0].x, uvs_before[0].y, uvs_before[1].x, uvs_before[1].y);
			float6 _zs_before(zs_before[0].x, zs_before[0].y, zs_before[0].z, zs_before[1].x, zs_before[1].y, zs_before[1].z);
			int2 _ids = make_int2(ids[0], ids[1]);
			
			#pragma unroll
			for (int s = 0; s < WINDOW_SIZE; ++s) 
			{
				if (_uvzs.z < sort_uvzs[s].z) 
				{
					swap(global_id , sort_global_ids[s]);
					swap(_indices, sort_indices[s]);
					swap(_uvzs, sort_uvzs[s]);
					swap(_uvs_before, sort_uvs_before[s]);
					swap(_zs_before, sort_zs_before[s]);
					swap(_ids, sort_ids[s]);
				}
			}
			++sort_num;
#else

			contributor++;
			float weight = alpha * T;
			T *= 1 - alpha;
			
			float dL_dalpha = 0.0f;
			o += weight;
			dL_dalpha += dL_dO * (1 - O);

			const float* t_features_ptr = t_features + global_id*S;
			float* dL_dt_features_ptr = dL_dt_features + global_id*S;
			for (int ch = 0; ch < S; ch++){
				float fea = t_features_ptr[ch];
				f[ch] += weight * fea;
				atomicAdd(&(dL_dt_features_ptr[ch]), weight * dL_dF[ch]);
				dL_dalpha += dL_dF[ch] * (T * fea - (F[ch] - f[ch]));
			}

			dL_dalpha /= max(0.00001f, 1.0f - alpha);

			float3 dL_duvz_0={0,0,0}, dL_duvz_1={0,0,0};

			if (alpha_before >= 0.0f){
				float dL_dsdf_prev_sig = dL_dalpha * (max(sdf_next_sig,0.00001f) / (max(sdf_prev_sig,0.00001f) * max(sdf_prev_sig,0.00001f)));
				float dL_dsdf_next_sig = dL_dalpha * (-1.0f / max(sdf_prev_sig,0.00001f));

				float dL_dsdf_prev = dL_dsdf_prev_sig * sdf_prev_sig * (1 - sdf_prev_sig);
				float dL_dsdf_next = dL_dsdf_next_sig * sdf_next_sig * (1 - sdf_next_sig);

				float* dL_dsdf_ptr = dL_dt_sdfs_var + global_id * 4;
				atomicAdd(&(dL_dsdf_ptr[indices_0.x]), dL_dsdf_prev * (1-uvz_0.x-uvz_0.y));
				atomicAdd(&(dL_dsdf_ptr[indices_0.y]), dL_dsdf_prev * uvz_0.x);
				atomicAdd(&(dL_dsdf_ptr[indices_0.z]), dL_dsdf_prev * uvz_0.y);

				atomicAdd(&(dL_dsdf_ptr[indices_1.x]), dL_dsdf_next * (1-uvz_1.x-uvz_1.y));
				atomicAdd(&(dL_dsdf_ptr[indices_1.y]), dL_dsdf_next * uvz_1.x);
				atomicAdd(&(dL_dsdf_ptr[indices_1.z]), dL_dsdf_next * uvz_1.y);

				dL_duvz_0.x += dL_dsdf_prev * (sdf_ptr[indices_0.y] - sdf_ptr[indices_0.x]);
				dL_duvz_0.y += dL_dsdf_prev * (sdf_ptr[indices_0.z] - sdf_ptr[indices_0.x]);

				dL_duvz_1.x += dL_dsdf_next * (sdf_ptr[indices_1.y] - sdf_ptr[indices_1.x]);
				dL_duvz_1.y += dL_dsdf_next * (sdf_ptr[indices_1.z] - sdf_ptr[indices_1.x]);
			}

			float2 uvs_before_0 = uvs_before[0], dL_duvs_before_0;
			float3 zs_before_0 = zs_before[0], dL_dzs_before_0;
			projection_correct_backward(uvs_before_0.x, uvs_before_0.y, zs_before_0.x, zs_before_0.y, zs_before_0.z,
				dL_duvz_0, dL_duvs_before_0.x, dL_duvs_before_0.y, dL_dzs_before_0.x, dL_dzs_before_0.y, dL_dzs_before_0.z);

			int id_0 = ids[0];
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 3]), dL_duvs_before_0.x * pixf.x);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 4]), dL_duvs_before_0.x * pixf.y);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 5]), dL_duvs_before_0.x);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 6]), dL_duvs_before_0.y * pixf.x);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 7]), dL_duvs_before_0.y * pixf.y);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_0 * 9 + 8]), dL_duvs_before_0.y);

			atomicAdd(&(dL_dt_depths[4 * global_id + indices_0.x]), dL_dzs_before_0.x);
			atomicAdd(&(dL_dt_depths[4 * global_id + indices_0.y]), dL_dzs_before_0.y);
			atomicAdd(&(dL_dt_depths[4 * global_id + indices_0.z]), dL_dzs_before_0.z);

			float2 uvs_before_1 = uvs_before[1], dL_duvs_before_1;
			float3 zs_before_1 = zs_before[1], dL_dzs_before_1;
			projection_correct_backward(uvs_before_1.x, uvs_before_1.y, zs_before_1.x, zs_before_1.y, zs_before_1.z,
				dL_duvz_1, dL_duvs_before_1.x, dL_duvs_before_1.y, dL_dzs_before_1.x, dL_dzs_before_1.y, dL_dzs_before_1.z);
			
			int id_1 = ids[1];
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 3]), dL_duvs_before_1.x * pixf.x);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 4]), dL_duvs_before_1.x * pixf.y);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 5]), dL_duvs_before_1.x);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 6]), dL_duvs_before_1.y * pixf.x);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 7]), dL_duvs_before_1.y * pixf.y);
			atomicAdd(&(dL_dt_A_inv[36 * global_id + id_1 * 9 + 8]), dL_duvs_before_1.y);

			atomicAdd(&(dL_dt_depths[4 * global_id + indices_1.x]), dL_dzs_before_1.x);
			atomicAdd(&(dL_dt_depths[4 * global_id + indices_1.y]), dL_dzs_before_1.y);
			atomicAdd(&(dL_dt_depths[4 * global_id + indices_1.z]), dL_dzs_before_1.z);

			if (T < 0.0001f){ 
				done = true;
			}
#endif
		}
	}
#ifdef RESORT
	if (!done) {
		while (sort_num > 0)
			blend_one();
	}
#endif
}

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
	bool debug)
{
	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
	// cudaEventRecord(start);
	// float milliseconds = 0;

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	char* scanning_space;
	uint32_t* point_offsets;
    cudaMalloc((void**)&scanning_space, P * sizeof(char));
    cudaMalloc((void**)&point_offsets, P * sizeof(uint32_t));

	size_t temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, point_offsets, point_offsets, P);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(scanning_space, temp_storage_bytes, tiles_touched, point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		depths_to_sort,
		point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		visibility_filter,
		rect,
		tiles_touched,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			ranges);
	CHECK_CUDA(, debug)


	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&milliseconds, start, stop);
	// std::cout << "num_rendered: " << num_rendered << "\n";
	// std::cout << "before: " << milliseconds << " ms\n";
	// cudaEventRecord(start);


    RenderForwardCUDAKernel<MAX_CHANNELS> <<<tile_grid, block >> > (
		S,
		ranges,
		binningState.point_list,
		width, height,
		t_A_inv,
		t_features,
		t_depths,
		t_sdfs_var,
		t_proj_2D_min_max,
		contrib,
		alpha, 
		feature, 
		error_map,
		alpha_threshold);
	
	cudaFree(scanning_space);
	cudaFree(point_offsets);

	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&milliseconds, start, stop);
	// std::cout << "after: " << milliseconds << " ms\n";
	// cudaEventDestroy(start);
	// cudaEventDestroy(stop);

	return num_rendered;
}



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
	float* dL_dt_features,
	float* dL_dt_depths,
	float* dL_dt_sdfs_var,
	float alpha_threshold,
	bool debug)
{
	BinningState binningState = BinningState::fromChunk(binning_buffer, num_rendered);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	RenderBackwardCUDAKernel<MAX_CHANNELS> <<<tile_grid, block >>>(
		S,
		ranges,
		binningState.point_list,
		width, height,
		t_A_inv,
		t_features,
		t_depths,
		t_sdfs_var,
		t_proj_2D_min_max,
		alpha,
		feature,
		contrib,
		dL_dalpha,
		dL_dfeature,
		dL_dt_A_inv,
		dL_dt_features,
		dL_dt_depths,
		dL_dt_sdfs_var,
		alpha_threshold
	);
}