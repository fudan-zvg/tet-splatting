import os
import torch
from torch.utils.cpp_extension import load
from typing import NamedTuple
parent_dir = "TetrahedronRender"
_C = load(
    name='TetrahedronRender_ext',
    extra_cflags=["-O3"],
    extra_include_paths=[
        os.path.join(parent_dir, "include"),
        os.path.join(parent_dir, "third_party/glm/"),
    ],
    sources=[
        os.path.join(parent_dir, "src", "preprocess.cu"),
        os.path.join(parent_dir, "src", "render.cu"),
        os.path.join(parent_dir, "src", "visible_tet.cu"),
        os.path.join(parent_dir, "src", "tet_gradients.cu"),
        os.path.join(parent_dir, "src", "ext.cpp"),
    ],
    verbose=True)

class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")
        

class TetrahedronRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    alpha_threshold: float
    
class _TetrahedronRender(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        depths_to_sort,  # used for sorting 
        t_A_inv,
        t_normals,
        t_depths,
        t_sdfs_var, 
        t_proj_2D_min_max,
        rect,
        tiles_touched,
        visibility_filter,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            depths_to_sort, 
            t_A_inv, 
            t_normals,
            t_depths,
            t_sdfs_var, 
            t_proj_2D_min_max,
            rect,
            tiles_touched,
            visibility_filter,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.alpha_threshold,
            False
        )
        num_rendered, contrib, alpha, normal, error_map, ranges, binningBuffer = _C.render_forward(*args)
        
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(contrib, t_A_inv, t_normals, t_depths, t_sdfs_var, t_proj_2D_min_max, alpha, normal, ranges, binningBuffer)
        return contrib, alpha, normal, error_map

    @staticmethod
    def backward(ctx, dL_dcontrib, dL_dalpha, dL_dnormal, dL_derror_map):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        contrib, t_A_inv, t_normals, t_depths, t_sdfs_var, t_proj_2D_min_max, alpha, normal, ranges, binningBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            t_A_inv,
            t_normals,
            t_depths,
            t_sdfs_var,
            t_proj_2D_min_max,
            alpha,
            normal,
            contrib,
            dL_dalpha, 
            dL_dnormal, 
            num_rendered,
            ranges,
            binningBuffer,
            raster_settings.alpha_threshold,
            False,
        )
        dL_dt_A_inv, dL_dt_normals, dL_dt_depths, dL_dt_sdfs_var = _C.render_backward(*args)
        grads = (
            None,  # depth_to_sort
            torch.nan_to_num(dL_dt_A_inv, 0.),
            torch.nan_to_num(dL_dt_normals, 0.),
            torch.nan_to_num(dL_dt_depths, 0.),
            torch.nan_to_num(dL_dt_sdfs_var, 0.),
            None,  # grad_t_proj_2D_min_max,
            None,  # grad_rect,
            None,  # grad_tiles_touched,
            None,  # grad_visibility_filter,
            None,  # raster_settings
        )
        return grads


class _TetrahedronPreprocess(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vertices,
        tet_indices,
        proj,
        w2c,
        height, 
        width,
    ):

        args = (
            vertices,
            tet_indices.int(),
            proj,
            w2c,
            height,
            width,
        )
        t_A_inv, depths, depths_to_sort, t_proj_2D_min_max, rect, tiles_touched, visibility_filter, depths, v_2D = _C.preprocess_forward(*args)
        if torch.isinf(t_A_inv).any():
            import pdb;pdb.set_trace()
        ctx.height = height
        ctx.width = width
        ctx.save_for_backward(vertices, tet_indices, proj, w2c, t_A_inv, visibility_filter)
        return t_A_inv, depths, depths_to_sort, t_proj_2D_min_max, rect, tiles_touched, visibility_filter

    @staticmethod
    def backward(ctx, dL_dt_A_inv, dL_ddepths, dL_ddepths_to_sort, dL_dt_proj_2D_min_max, dL_drect, 
                 dL_dtiles_touched, dL_dvisibility_filter):

        vertices, tet_indices, proj, w2c, t_A_inv, visibility_filter = ctx.saved_tensors
        height = ctx.height
        width = ctx.width
        # Restructure args as C++ method expects them
        args = (
            vertices,
            tet_indices.int(),
            proj,
            w2c,
            height,
            width,
            t_A_inv,
            visibility_filter,
            dL_dt_A_inv,
            dL_ddepths
        )
        dL_dvertices = _C.preprocess_backward(*args)
        if torch.isnan(dL_dvertices).any():
            print("dL_dvertices has NaNs")
            import pdb;pdb.set_trace()
            torch.isinf(t_A_inv)
        grads = (
            torch.nan_to_num(dL_dvertices, 0.),  # vertices
            None,
            None,
            None,
            None,
            None
        )
        return grads

def preprocess_tet(vertices, tet_indices, proj, w2c, height, width, use_python=False):
    if use_python:
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
        v_hom = vertices_homo @ proj.T
        v_proj = v_hom[:, :3] / (v_hom[:, 3:] + 0.0000001)
        v_view = vertices @ w2c.T[:3, :3] + w2c[:3, 3]
        v_2D = torch.stack([
            ((v_proj[..., 0] + 1.0) * width - 1) * 0.5,
            ((v_proj[..., 1] + 1.0) * height - 1) * 0.5,
        ], dim=-1)
        v_2D_homo = torch.cat([v_2D, torch.ones_like(v_2D[..., :1])], dim=-1)
        depths = -v_view[:, 2:3]
        t_proj_2D = v_2D[tet_indices]
        t_proj_2D_homo = v_2D_homo[tet_indices]
        t_A = t_proj_2D_homo[:, [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]].transpose(-1, -2)
        t_A_inv = t_A.inverse()
        t_depths = depths[tet_indices]
        with torch.no_grad():
            depths_to_sort = t_depths.mean(dim=1)
            BLOCK_X = BLOCK_Y = 16
            visibility_filter = t_depths[..., 0].min(dim=-1).values > 0.01
            grid = [int((width + BLOCK_X - 1) / BLOCK_X), int((height + BLOCK_Y - 1) / BLOCK_Y)]
            t_proj_2D_min = t_proj_2D.min(dim=1)[0].floor()
            t_proj_2D_max = t_proj_2D.max(dim=1)[0].ceil()
            t_proj_2D_min_max = torch.cat([t_proj_2D_min, t_proj_2D_max], dim=-1).int()
            rect_min = torch.stack([
                (t_proj_2D_min[:, 0] / BLOCK_X).int().clamp(0, grid[0]),
                (t_proj_2D_min[:, 1] / BLOCK_Y).int().clamp(0, grid[1]),
            ], dim=-1).int()
            rect_max = torch.stack([
                ((t_proj_2D_max[:, 0] + BLOCK_X - 1) / BLOCK_X).int().clamp(0, grid[0]),
                ((t_proj_2D_max[:, 1] + BLOCK_Y - 1) / BLOCK_Y).int().clamp(0, grid[1]),
            ], dim=-1).int()
            rect = torch.cat([rect_min, rect_max], dim=-1)
            tiles_touched = (rect_max[:, 0] - rect_min[:, 0]) * (rect_max[:, 1] - rect_min[:, 1])
            tiles_touched[~visibility_filter] = 0
            visibility_filter[tiles_touched == 0] = False
        return t_A_inv, depths, depths_to_sort, t_proj_2D_min_max, rect, tiles_touched, visibility_filter
    else:
        return _TetrahedronPreprocess.apply(vertices, tet_indices, proj, w2c, height, width)
        

@torch.no_grad()
def get_visible_tet(sdfs_var, tet_indices, alpha_threshold, use_python=False):
    if use_python:
        t_sdfs_var = sdfs_var[tet_indices]
        sdf_var_max = t_sdfs_var.max(dim=1).values
        sdf_var_min = t_sdfs_var.min(dim=1).values
        alpha_max = (1 - torch.sigmoid(sdf_var_min).clamp_min(1e-5) / torch.sigmoid(sdf_var_max).clamp_min(1e-5)).clamp_min(0).squeeze(-1)
        t_mask = alpha_max > alpha_threshold
        return t_mask
    else:
        return _C.get_visible_tet(sdfs_var, tet_indices.int(), alpha_threshold)

class _GetTetGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tet_indices, sdfs, vertices):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            tet_indices, 
            sdfs, 
            vertices,
        )
        t_gradients, A_inv = _C.get_tet_gradients_forward(*args)
        
        # Keep relevant tensors for backward
        ctx.save_for_backward(tet_indices, sdfs, vertices, A_inv)
        return t_gradients

    @staticmethod
    def backward(ctx, dL_dt_gradients):
        tet_indices, sdfs, vertices, A_inv = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            tet_indices,
            sdfs,
            vertices,
            A_inv,
            dL_dt_gradients,
        )
        dL_dsdfs, dL_dvertices = _C.get_tet_gradients_backward(*args)
        grads = (
            None,  # tet_indices
            dL_dsdfs,
            dL_dvertices,
        )
        return grads

def get_tet_gradients(tet_indices, sdfs, vertices, use_python=False):
    if use_python:
        t_sdfs = sdfs[tet_indices].clone().contiguous()
        t_vertices = vertices[tet_indices]
        t_vertices_homo = torch.cat([t_vertices, torch.ones_like(t_vertices[..., :1])], dim=-1)
        A_inv = t_vertices_homo.inverse().clone().contiguous()
        sdf_grad = (A_inv @ t_sdfs).squeeze(-1)[:, :3]
        # (A_inv.transpose(-1,-2)@grad_homo[..., None])
        # (grad_homo[..., None]@t_sdfs.transpose(-1,-2))
        # grad_A_inv = (grad_homo[..., None]@t_sdfs.transpose(-1,-2))
        # (-A_inv[0].T * A_inv.grad[0] * A_inv[0]).T
        # grad = torch.randn_like(sdf_grad)
        # grad_homo = torch.cat([grad, torch.zeros_like(grad[..., :1])], dim=-1)
        # sdf_grad.backward(gradient=grad, retain_graph=True)
        # -(A_inv[0].T@A_inv.grad[0]@A_inv[0].T)
        # grad
        return sdf_grad
    else:
        sdf_grad = _GetTetGradients.apply(tet_indices.int(), sdfs, vertices)
        return _GetTetGradients.apply(tet_indices.int(), sdfs, vertices)


def render_tet(depths_to_sort, t_A_inv, t_normals, t_depths, t_sdfs_var, 
               t_proj_2D_min_max, rect, tiles_touched, 
               visibility_filter, raster_settings):
    return _TetrahedronRender.apply(
            depths_to_sort,
            t_A_inv,
            t_normals,
            t_depths,
            t_sdfs_var, 
            t_proj_2D_min_max,
            rect,
            tiles_touched,
            visibility_filter,
            raster_settings,
        )