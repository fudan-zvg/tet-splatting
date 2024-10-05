import nerfacc
import pdb
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from einops import repeat

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.transfer import rotate_y
from threestudio.utils.typing import *
from custom_functions import TetrahedronRasterizationSettings, preprocess_tet, render_tet
    

def c2wtow2c(c2w):
    """transfer camera2world to world2camera matrix"""

    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0

    return w2c


@threestudio.register("tetsplatting-rasterizer")
class TeTSplattingRasterizerDMTet(Rasterizer):
    """
    using different bg for sd and colors
    """

    @dataclass
    class Config(VolumeRenderer.Config):
        camera_space: bool = False
        near_distance: float = 1.732
        
        mv_bg_colors: str = "blue"
        sd_bg_colors: str = "blue"

    cfg: Config

    @staticmethod
    def world2camera(normal, w2c):
        rotate: Float[Tensor, "B 4 4"] = w2c[..., :3, :3]
        camera_normal = normal @ rotate.permute(0, 2, 1)
        # pixel space flip axis so we need built negative y-axis normal
        flip_x = torch.eye(3).to(w2c)
        flip_x[0, 0] = -1

        camera_normal = camera_normal @ flip_x[None, ...]

        return camera_normal

    def obtain_bg_colors(self, gb_normal, opacity, bg_colors="blue"):
        if bg_colors == "blue":
            _a = torch.zeros_like(gb_normal)
            _a[..., 2] = 1.0
            _a = (_a + 1.0) / 2.0
        elif bg_colors == "black":
            _a = torch.zeros_like(gb_normal)
        elif bg_colors == "white":
            _a = torch.ones_like(gb_normal)
        else:
            raise NotImplementedError

        _b = (gb_normal + 1.0) / 2.0
        gb_normal_aa = torch.lerp(_a, _b, opacity)
        return gb_normal_aa

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(
        self,
        c2w: Float[Tensor, "B 4 4"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        alpha_threshold=1/255,
        use_python: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]

        raster_settings = TetrahedronRasterizationSettings(
            image_height=height,
            image_width=width,
            alpha_threshold=alpha_threshold,
        )
        
        states = self.geometry.states
        vertices = states["vertices"]
        tet_indices = states["tet_indices"]
        t_sdf_var = states["t_sdf_var"]
        t_normals = states["t_normals"]
        t_mask = states["t_mask"]
        t_gradients = states["t_gradients"]
        inv_s = states["inv_s"]
        
        t_normals = repeat(t_normals, "n c -> b n c", b=batch_size)
        if self.cfg.camera_space:
            w2c = c2wtow2c(c2w)
            t_normals = self.world2camera(t_normals, w2c)
            
        results = []
        for b in range(batch_size):
            w2c = c2w[b].inverse()
            t_A_inv, depths, depths_to_sort, t_proj_2D_min_max, rect, tiles_touched, visibility_filter = preprocess_tet(vertices, tet_indices, mvp_mtx[b], w2c, height, width, use_python=use_python)

            t_depths = depths[tet_indices]
            t_features = torch.zeros_like(t_depths)
            
            t_features = torch.cat([t_normals[b], t_depths.mean(dim=-2)], dim=-1)
            contrib, alpha, feature, error_map = render_tet(
                depths_to_sort,
                t_A_inv,
                t_features,
                t_depths,
                t_sdf_var, 
                t_proj_2D_min_max,
                rect,
                tiles_touched,
                visibility_filter,
                raster_settings,
            )
            
            alpha = alpha.permute(1, 2, 0)
            feature = feature.permute(1, 2, 0)
            feature = feature / alpha.clamp_min(1e-5)
            normal, depth = feature.split([3, 1], dim=-1)
            normal = F.normalize(normal, dim=-1)
            contrib = contrib.permute(1, 2, 0)
            result = {
                "opacity": alpha,
                "depth": depth,
                "contrib": contrib,
                "normal": normal,
            }            
            results.append(result)
        out = {k: torch.stack([res[k] for res in results], dim=0) for k in result}
        
        depth = out["depth"] 
        opacity = out["opacity"] 
        normal = out["normal"]
        
        mv_normal_aa = self.obtain_bg_colors(normal, opacity, self.cfg.mv_bg_colors)
        sd_normal_aa = self.obtain_bg_colors(normal, opacity, self.cfg.sd_bg_colors)
        
        out.update(
            {
                "comp_normal": mv_normal_aa.clamp(0, 1),
                "sd_comp_normal": sd_normal_aa.clamp(0, 1),
                "opacity": opacity,
            }
        )  # in [0, 1]

        # mv_disparity
        # 1. comnpute disparity_free scale
        near_d = kwargs["camera_distances"] - self.cfg.near_distance
        far_d = kwargs["camera_distances"] + self.cfg.near_distance

        near_d = near_d[:, None, None, None].expand_as(depth)
        far_d = far_d[:, None, None, None].expand_as(depth)
        depth[depth > far_d] = far_d[depth > far_d]

        mv_disparity = (far_d - depth) / (2 * self.cfg.near_distance)  # 2 * sqrt(3)

        mv_disparity_aa = torch.lerp(
            torch.zeros_like(mv_disparity), mv_disparity, opacity
        )
        out.update({"mv_disparity": mv_disparity_aa.clamp(0, 1)})  # in [0, 1]
        
        out.update({
            "gradients": t_gradients,  
            "filter" : t_mask,
            "inv_s" : inv_s,
            "N": t_sdf_var.shape[0],
        })
        return out
