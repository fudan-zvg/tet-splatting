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

#include <torch/extension.h>
#include "render.h"
#include "preprocess.h"
#include "visible_tet.h"
#include "tet_gradients.h"
#include "rasterizer_impl.h"
#include "config.h"
#include <cuda_runtime_api.h>

std::function<char*(size_t N)> resizeFunc(torch::Tensor& t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long)N});
    return reinterpret_cast<char*>(t.contiguous().data_ptr());
  };
  return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderForward(
  const torch::Tensor& depths_to_sort,
  const torch::Tensor& t_A_inv,
  const torch::Tensor& t_features,
  const torch::Tensor& t_depths,
  const torch::Tensor& t_sdfs_var,
  const torch::Tensor& t_proj_2D_min_max,
  const torch::Tensor& rect,
  const torch::Tensor& tiles_touched,
  const torch::Tensor& visibility_filter,
  const int image_height,
  const int image_width,
  const float alpha_threshold,
	const bool debug)
{
  const int P = t_features.size(0);
  const int S = t_features.size(1);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = t_features.options().dtype(torch::kInt32);
  auto float_opts = t_features.options().dtype(torch::kFloat32);

  torch::Tensor alpha = torch::zeros({1, H, W}, float_opts);
  torch::Tensor feature = torch::zeros({S, H, W}, float_opts);
  torch::Tensor contrib = torch::zeros({1, H, W}, int_opts);
  torch::Tensor ranges = torch::zeros({H, W, 2}, int_opts);
  torch::Tensor error_map = torch::zeros({1, H, W}, float_opts);
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> binningFunc = resizeFunc(binningBuffer);

  int rendered = 0;
  if(P != 0)
  {
    rendered = RenderForwardCUDA(
      P, S, W, H,
      depths_to_sort.contiguous().data_ptr<float>(),
      t_A_inv.contiguous().data_ptr<float>(),
      t_features.contiguous().data_ptr<float>(), 
      t_depths.contiguous().data_ptr<float>(),
      t_sdfs_var.contiguous().data_ptr<float>(),
      t_proj_2D_min_max.contiguous().data_ptr<int>(),
      rect.contiguous().data_ptr<int>(),
      tiles_touched.contiguous().data_ptr<int>(),
      visibility_filter.contiguous().data_ptr<bool>(),
      binningFunc,
      contrib.contiguous().data_ptr<int>(),
      alpha.contiguous().data_ptr<float>(),
      feature.contiguous().data_ptr<float>(),
      error_map.contiguous().data_ptr<float>(),
      (int2*)ranges.contiguous().data_ptr<int>(),
      alpha_threshold,
      debug);
  }
  return std::make_tuple(rendered, contrib, alpha, feature, error_map, ranges, binningBuffer);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RenderBackward(
  const torch::Tensor& t_A_inv,
  const torch::Tensor& t_features,
  const torch::Tensor& t_depths,
  const torch::Tensor& t_sdfs_var,
  const torch::Tensor& t_proj_2D_min_max,
  const torch::Tensor& alpha,
  const torch::Tensor& feature,
  const torch::Tensor& contrib,
  const torch::Tensor& dL_dalpha,
  const torch::Tensor& dL_dfeature,
	const int num_rendered,
	const torch::Tensor& ranges,
	const torch::Tensor& binningBuffer,
  const float alpha_threshold,
	const bool debug) 
{
  const int P = t_features.size(0);
  const int S = t_features.size(1);
  const int H = dL_dfeature.size(1);
  const int W = dL_dfeature.size(2);
  
  auto float_opts = t_features.options().dtype(torch::kFloat32);
  torch::Tensor dL_dt_A_inv = torch::zeros({P, 4, 3, 3}, float_opts);
  torch::Tensor dL_dt_features = torch::zeros({P, S}, float_opts);
  torch::Tensor dL_dt_depths = torch::zeros({P, 4, 1}, float_opts);
  torch::Tensor dL_dt_sdfs_var = torch::zeros({P, 4, 1}, float_opts);
  
  if(P != 0)
  {  
	  RenderBackwardCUDA(
      P, S, num_rendered, W, H, 
      t_A_inv.contiguous().data_ptr<float>(),
      t_features.contiguous().data_ptr<float>(),
      t_depths.contiguous().data_ptr<float>(),
      t_sdfs_var.contiguous().data_ptr<float>(),
      t_proj_2D_min_max.contiguous().data_ptr<int>(),
      alpha.contiguous().data_ptr<float>(),
      feature.contiguous().data_ptr<float>(),
      contrib.contiguous().data_ptr<int>(),
      (int2*)ranges.contiguous().data_ptr<int>(),
      reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
      dL_dalpha.contiguous().data_ptr<float>(),
      dL_dfeature.contiguous().data_ptr<float>(),
      dL_dt_A_inv.contiguous().data_ptr<float>(),
      dL_dt_features.contiguous().data_ptr<float>(),
      dL_dt_depths.contiguous().data_ptr<float>(),
      dL_dt_sdfs_var.contiguous().data_ptr<float>(),
      alpha_threshold,
      debug
    );
  }

  return std::make_tuple(dL_dt_A_inv, dL_dt_features, dL_dt_depths, dL_dt_sdfs_var);
}

torch::Tensor GetVisibleTet(
  const torch::Tensor& sdfs_var,
  const torch::Tensor& tet_indices,
  const float alpha_threshold)
{
  const int P = tet_indices.size(0);
  torch::Tensor t_mask = torch::zeros({P}, tet_indices.options().dtype(torch::kBool));
  
  if(P != 0)
  {
    GetVisibleTetCUDA(
      P,
      sdfs_var.contiguous().data_ptr<float>(),
      tet_indices.contiguous().data_ptr<int>(),
      t_mask.contiguous().data_ptr<bool>(),
      alpha_threshold);
  }
  return t_mask;
}

std::tuple<torch::Tensor, torch::Tensor>
GetTetGradientsForward(
  const torch::Tensor& tet_indices,
  const torch::Tensor& sdfs,
  const torch::Tensor& vertices
){
  const int P = tet_indices.size(0);
  torch::Tensor t_gradients = torch::zeros({P, 3}, sdfs.options());
  torch::Tensor A_inv = torch::zeros({P, 4, 4}, sdfs.options());
  if(P != 0)
  {
    GetTetGradientsForwardCUDA(
      P,
      tet_indices.contiguous().data_ptr<int>(),
      sdfs.contiguous().data_ptr<float>(),
      vertices.contiguous().data_ptr<float>(),
      t_gradients.contiguous().data_ptr<float>(),
      A_inv.contiguous().data_ptr<float>()
      );
  }
  return std::make_tuple(t_gradients, A_inv);
}

std::tuple<torch::Tensor, torch::Tensor>
GetTetGradientsBackward(
  const torch::Tensor& tet_indices,
  const torch::Tensor& sdfs,
  const torch::Tensor& vertices,
  const torch::Tensor& A_inv,
  const torch::Tensor& dL_dt_gradients
){
  const int P = tet_indices.size(0);
  const int V = sdfs.size(0);
  torch::Tensor dL_dsdfs = torch::zeros({V, 1}, sdfs.options());
  torch::Tensor dL_dvertices = torch::zeros({V, 3}, sdfs.options());
  if(P != 0)
  {
    GetTetGradientsBackwardCUDA(
      P,
      tet_indices.contiguous().data_ptr<int>(),
      sdfs.contiguous().data_ptr<float>(),
      vertices.contiguous().data_ptr<float>(),
      A_inv.contiguous().data_ptr<float>(),
      dL_dt_gradients.contiguous().data_ptr<float>(),
      dL_dsdfs.contiguous().data_ptr<float>(),
      dL_dvertices.contiguous().data_ptr<float>()
      );
  }
  return std::make_tuple(dL_dsdfs, dL_dvertices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PreprocessForward(
  const torch::Tensor& vertices,
  const torch::Tensor& tet_indices,
  const torch::Tensor& proj,
  const torch::Tensor& w2c,
  const int image_height,
  const int image_width)
{
  const int P = tet_indices.size(0);
  const int V = vertices.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = vertices.options().dtype(torch::kInt32);
  auto float_opts = vertices.options().dtype(torch::kFloat32);
  auto bool_opts = vertices.options().dtype(torch::kBool);

  torch::Tensor depths = torch::zeros({V, 1}, float_opts);
  torch::Tensor v_2Ds = torch::zeros({V, 2}, float_opts);
  torch::Tensor t_A_inv = torch::zeros({P, 4, 3, 3}, float_opts);
  torch::Tensor depths_to_sort = torch::zeros({P, 1}, float_opts);
  torch::Tensor t_proj_2D_min_max = torch::zeros({P, 4}, int_opts);
  torch::Tensor rect = torch::zeros({P, 4}, int_opts);
  torch::Tensor tiles_touched = torch::zeros({P}, int_opts);
  torch::Tensor visibility_filter = torch::zeros({P}, bool_opts);

  if(P != 0)
  {
    PreprocessForwardCUDA(
      P, V, H, W, 
      vertices.contiguous().data_ptr<float>(),
      tet_indices.contiguous().data_ptr<int>(),
      proj.contiguous().data_ptr<float>(),
      w2c.contiguous().data_ptr<float>(),
      depths.contiguous().data_ptr<float>(),
      v_2Ds.contiguous().data_ptr<float>(),
      t_A_inv.contiguous().data_ptr<float>(),
      depths_to_sort.contiguous().data_ptr<float>(),
      t_proj_2D_min_max.contiguous().data_ptr<int>(),
      rect.contiguous().data_ptr<int>(),
      tiles_touched.contiguous().data_ptr<int>(),
      visibility_filter.contiguous().data_ptr<bool>()
      );
  }
  return std::make_tuple(t_A_inv, depths, depths_to_sort, t_proj_2D_min_max, rect, tiles_touched, visibility_filter, depths, v_2Ds);
}

torch::Tensor
PreprocessBackward(
  const torch::Tensor& vertices,
  const torch::Tensor& tet_indices,
  const torch::Tensor& proj,
  const torch::Tensor& w2c,
  const int image_height,
  const int image_width,
  const torch::Tensor& t_A_inv,
  const torch::Tensor& visibility_filter,
  const torch::Tensor& dL_dt_A_inv,
  const torch::Tensor& dL_ddepths
  )
{
  const int P = tet_indices.size(0);
  const int V = vertices.size(0);
  const int H = image_height;
  const int W = image_width;

  auto float_opts = vertices.options().dtype(torch::kFloat32);
  torch::Tensor dL_dvertices = torch::zeros({V, 3}, float_opts);
  torch::Tensor dL_dv_2Ds = torch::zeros({V, 2}, float_opts);

  if(P != 0)
  {
    PreprocessBackwardCUDA(
      P, V, H, W, 
      vertices.contiguous().data_ptr<float>(),
      tet_indices.contiguous().data_ptr<int>(),
      proj.contiguous().data_ptr<float>(),
      w2c.contiguous().data_ptr<float>(),
      t_A_inv.contiguous().data_ptr<float>(),
      visibility_filter.contiguous().data_ptr<bool>(),
      dL_dt_A_inv.contiguous().data_ptr<float>(),
      dL_ddepths.contiguous().data_ptr<float>(),
      dL_dvertices.contiguous().data_ptr<float>(),
      dL_dv_2Ds.contiguous().data_ptr<float>()
      );
  }
  return dL_dvertices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_forward", &RenderForward);
  m.def("render_backward", &RenderBackward);
  m.def("get_visible_tet", &GetVisibleTet);
  m.def("get_tet_gradients_forward", &GetTetGradientsForward);
  m.def("get_tet_gradients_backward", &GetTetGradientsBackward);
  m.def("preprocess_forward", &PreprocessForward);
  m.def("preprocess_backward", &PreprocessBackward);
}