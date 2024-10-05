import bisect
import cv2
import numpy as np
import os
import pdb
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from einops import repeat
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@dataclass
class DirectionConfig:
    name: str
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]


@dataclass
class ZeroDirectionConfig:
    name: str
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]


def shift_azimuth_deg(azimuth: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    # shift azimuth angle (in degrees), to [-180, 180]
    return (azimuth + 180) % 360 - 180


class _Base_ND_TeTSplatting(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        latent_steps: int = 1000
        texture: bool = False
        camera_space: bool = False
        is_refine: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()

        # view idx
        self.directions = [
            DirectionConfig(
                "side",
                lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
            ),
            DirectionConfig(
                "front",
                lambda ele, azi, dis: (
                    shift_azimuth_deg(azi) > -self.cfg.front_threshold
                )
                & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
            ),
            DirectionConfig(
                "back",
                lambda ele, azi, dis: (
                    shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                )
                | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
            ),
        ]
        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        self._val_step = 0  # using to validation_step

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_rgb=self.cfg.texture)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            if not self.cfg.is_refine:
                self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        loss = 0.0

        out = self(batch)
        prompt_utils = self.prompt_processor()

        if not self.cfg.texture:  # geometry training
            if self.true_global_step < self.cfg.latent_steps:
                normal = out["comp_normal"] * 2.0 - 1.0

                guidance_inp = torch.cat([normal, out["opacity"]], dim=-1)

                guidance_out = self.guidance(
                    {"comp_rgb": guidance_inp},
                    prompt_utils,
                    **batch,
                    rgb_as_latents=True,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                )
            else:
                guidance_inp = out["comp_normal"]
                guidance_out = self.guidance(
                    {"comp_rgb": guidance_inp},
                    prompt_utils,
                    **batch,
                    rgb_as_latents=False,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                )

            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
        else:  # texture training
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        if not batch["front"]["view_step"] == self._val_step:
            return
        front = batch["front"]
        front["height"] = batch["height"]
        front["width"] = batch["width"]

        out = self(front)

        self.save_image_grid(
            f"it{self.true_global_step:06d}-{front['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_disparity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
                if "comp_disparity" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["mv_disparity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
                if "mv_disparity" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        self._val_step += 1

    def test_step(self, batch, batch_idx):
        front = batch["front"]
        front["height"] = batch["height"]
        front["width"] = batch["width"]

        out = self(front)

        self.save_image_grid(
            f"it{self.true_global_step:06d}-test/{front['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["sd_comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["mv_disparity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
                if "mv_disparity" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        if self.trainer.global_rank == 0:
            self.save_img_ffmpeg(
                f"it{self.true_global_step:06d}-test.mp4",
                30,
                f"./",
                f"it{self.true_global_step:06d}-test",
            )


class _Base_ND_TeTSplatting_MV(_Base_ND_TeTSplatting):
    @dataclass
    class Config(_Base_ND_TeTSplatting.Config):
        # multiview-guidance
        mv_guidance_type: str = ""
        mv_guidance: dict = field(default_factory=dict)
        mv_prompt_processor_type: str = ""
        mv_prompt_processor: dict = field(default_factory=dict)
        debug: bool = False
        anneal_normal_stone: Optional[Any] = None

        use_python: bool = False
        alpha_threshold: float = 1 / 255
        step_start: int = 0
        
    cfg: Config

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, 
                                   render_rgb=self.cfg.texture,
                                   alpha_threshold=self.cfg.alpha_threshold,
                                   use_python=self.cfg.use_python)
        return {
            **render_out,
        }
        
    
    def sd_loss(self, out, batch):
        loss = 0.0
        prompt_utils = self.prompt_processor()

        if not self.cfg.texture:  # geometry training
            if self.true_global_step < self.cfg.latent_steps:
                normal = out["comp_normal"] * 2.0 - 1.0

                guidance_inp = torch.cat([normal, out["opacity"]], dim=-1)

                guidance_out = self.guidance(
                    {"comp_rgb": guidance_inp},
                    prompt_utils,
                    **batch,
                    rgb_as_latents=True,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                )
            else:
                guidance_inp = out["comp_normal"]
                guidance_out = self.guidance(
                    {"comp_rgb": guidance_inp},
                    prompt_utils,
                    **batch,
                    rgb_as_latents=False,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                )

            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)

            # anneal refine strategy
            if self.cfg.anneal_normal_stone is None:
                anneal_weights = 1.0 * self.C(self.cfg.loss.lambda_normal_consistency)
            else:
                anneal_idx = bisect.bisect_left(
                    self.cfg.anneal_normal_stone, self.global_step
                )
                anneal_weights = (10**anneal_idx) * self.C(
                    self.cfg.loss.lambda_normal_consistency
                )  # NOTE that different with

            loss += loss_normal_consistency * anneal_weights

        else:  # texture training
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"sd_loss": loss}

    def nd_mv_sd_loss(self, out, batch, batch_idx, guidance: str):
        """
        Args:
            guidance:  "mv_loss"
        """

        loss = 0.0
        prompt_utils = self.nd_mv_prompt_processor()

        if not self.cfg.texture:  # geometry training
            # need normalized first
            normal = out["comp_normal"] * 2.0 - 1.0
            disparity = out["mv_disparity"] * 2.0 - 1.0
            guidance_inp = torch.cat([normal, disparity], dim=-1)

            guidance_out = self.nd_mv_guidance(
                {"comp_rgb": guidance_inp},
                prompt_utils,
                **batch,
                rgb_as_latents=True,
                current_step_ratio=self.true_global_step / self.trainer.max_steps,
            )

        else:  # texture training
            guidance_inp = out["comp_rgb"]
            guidance_out = self.nd_mv_guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"nd_sd_loss": loss}

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss_dict = {}

        view_loss = self.nd_mv_sd_loss(out, batch, batch_idx, "mv_loss")
        loss_dict.update(view_loss)

        # Stablediffusion Loss
        sd_loss = self.sd_loss(out, batch)
        loss_dict.update(sd_loss)

        loss = 0.0

        for name, value in loss_dict.items():
            loss_weighted = value * self.C(self.cfg.loss[name])

            loss += loss_weighted

        return {"loss": loss}

    def on_fit_start(self) -> None:
        # build nd_mv guidnace
        if self.cfg.nd_guidance_type == "":
            self.nd_mv_guidance = None
        else:
            self.nd_mv_prompt_processor = threestudio.find(
                self.cfg.nd_prompt_processor_type
            )(self.cfg.nd_prompt_processor)
            self.nd_mv_guidance = threestudio.find(self.cfg.nd_guidance_type)(
                self.cfg.nd_guidance
            )
        # fine space
        super().on_fit_start()


@threestudio.register("nd-tetsplatting-mv-system")
class ND_TeTSplatting_MV(_Base_ND_TeTSplatting_MV):
    
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        super().on_train_batch_start(batch, batch_idx, unused)
        self.geometry.update_states(
            step=self.global_step+self.cfg.step_start, 
            alpha_threshold=self.cfg.alpha_threshold, 
            use_python=self.cfg.use_python)
    
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        with torch.no_grad():
            self.geometry.update_states(
                step=self.global_step+self.cfg.step_start, 
                alpha_threshold=self.cfg.alpha_threshold, 
                use_python=self.cfg.use_python)
    
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        with torch.no_grad():
            self.geometry.update_states(
                step=self.global_step+self.cfg.step_start, 
                alpha_threshold=self.cfg.alpha_threshold, 
                use_python=self.cfg.use_python)
    
    def nd_mv_sd_loss(self, out, batch, batch_idx, guidance: str):
        """
        Args:
            guidance:  "mv_loss"
        """

        loss = 0.0
        prompt_utils = self.nd_mv_prompt_processor()

        if not self.cfg.texture:  # geometry training
            # need normalized first
            normal = out["comp_normal"] * 2.0 - 1.0
            disparity = out["mv_disparity"] * 2.0 - 1.0
            guidance_inp = torch.cat([normal, disparity], dim=-1)

            scale = guidance_inp.shape[-3] // 64
            guidance_inp = F.avg_pool2d(guidance_inp.permute(0, 3, 1, 2), scale, scale).permute(0, 2, 3, 1)
            
            guidance_out = self.nd_mv_guidance(
                {"comp_rgb": guidance_inp},
                prompt_utils,
                **batch,
                rgb_as_latents=True,
                current_step_ratio=self.true_global_step / self.trainer.max_steps,
            )

        else:  # texture training
            guidance_inp = out["comp_rgb"]
            guidance_out = self.nd_mv_guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"nd_sd_loss": loss}

    def sd_loss(self, out, batch):
        loss = 0.0
        prompt_utils = self.prompt_processor()

        if not self.cfg.texture:  # geometry training
            if self.true_global_step < self.cfg.latent_steps:
                normal = out["sd_comp_normal"] * 2.0 - 1.0

                guidance_inp = torch.cat([normal, out["opacity"]], dim=-1)

                scale = guidance_inp.shape[-3] // 64
                guidance_inp = F.avg_pool2d(guidance_inp.permute(0, 3, 1, 2), scale, scale).permute(0, 2, 3, 1)
            
                guidance_out = self.guidance(
                    {"comp_rgb": guidance_inp},
                    prompt_utils,
                    **batch,
                    rgb_as_latents=True,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                )
            else:
                guidance_inp = out["sd_comp_normal"]
                guidance_out = self.guidance(
                    {"comp_rgb": guidance_inp},
                    prompt_utils,
                    **batch,
                    rgb_as_latents=False,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                )            
        else:  # texture training
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch, rgb_as_latents=False
            )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"sd_loss": loss}

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss_dict = {}

        view_loss = self.nd_mv_sd_loss(out, batch, batch_idx, "mv_loss")
        loss_dict.update(view_loss)

        # Stablediffusion Loss
        sd_loss = self.sd_loss(out, batch)
        loss_dict.update(sd_loss)

        loss = 0.0
        
        sdf_gradient = out['gradients']
        loss_eikonal = (sdf_gradient.norm(dim=-1) - 1).pow(2).mean()
        
        loss += self.C(self.cfg.loss.eikonal_loss) * loss_eikonal
        
        all_edges = self.geometry.states['tet_indices'][:, self.geometry.isosurface_helper.base_tet_edges].reshape(-1, 2)
        all_edges = self.geometry.isosurface_helper.sort_edges(all_edges).reshape(-1, 2)
        edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        edges = edges.long()
        edge_nrm: Float[Tensor, "Ne 2 3"] = self.geometry.states['v_normals'][edges]
        loss_normal_consistency = (1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)).mean()
        
        self.log("consistency", loss_normal_consistency, True)
        loss += self.C(self.cfg.loss.lambda_normal_consistency) * loss_normal_consistency

        self.log(f"eik", loss_eikonal.item(), True)
        self.log(f"inv_s", out['inv_s'].item(), True)
        self.log(f"filter", out['filter'].float().mean().item(), True)
        self.log(f"TET", out['N'], True)

        for name, value in loss_dict.items():
            loss_weighted = value * self.C(self.cfg.loss[name])

            loss += loss_weighted

        return {"loss": loss}
    
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        front = batch
        out = self(batch)
        # if not batch["front"]["view_step"] == self._val_step:
        #     return
        # front = batch["front"]
        # front["height"] = batch["height"]
        # front["width"] = batch["width"]
        # out = self(front)


        self.save_image_grid(
            f"it{self.true_global_step:06d}-{front['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["sd_comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["mv_disparity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
                if "mv_disparity" in out
                else []
            )+ (
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
                if "mv_disparity" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step:06d}-test",
            f"it{self.true_global_step:06d}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )


@threestudio.register("nd-tetsplatting-mv-texture-system")
class ND_TeTSplatting_MV_Texture(_Base_ND_TeTSplatting_MV):
    @dataclass
    class Config(_Base_ND_TeTSplatting_MV.Config):
        # multiview-guidance
        albedo_guidance_type: str = ""
        albedo_guidance: dict = field(default_factory=dict)
        albedo_prompt_processor_type: str = ""
        albedo_prompt_processor: dict = field(default_factory=dict)

    cfg: Config

    def sd_loss(self, out, batch):
        # stable-diffusion loss

        loss = 0.0
        prompt_utils = self.prompt_processor()

        # texture training
        guidance_inp = out["comp_rgb"]
        guidance_out = self.guidance(
            {"comp_rgb": guidance_inp},
            prompt_utils,
            **batch,
            rgb_as_latents=False,
            current_step_ratio=self.true_global_step / self.trainer.max_steps,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"sd_loss": loss}

    def albedo_mv_sd_loss(self, out, batch, batch_idx, guidance: str):
        """
        Args:
            for albedo loss
            guidance:  "mv_loss"
        """

        loss = 0.0
        prompt_utils = self.albedo_mv_prompt_processor()

        guidance_out = self.albedo_mv_guidance(
            out,
            prompt_utils,
            **batch,
            rgb_as_latents=False,
            current_step_ratio=self.true_global_step / self.trainer.max_steps,
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"mv_sd_loss": loss}

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss_dict = {}

        if not self.albedo_mv_guidance == None:
            view_loss = self.albedo_mv_sd_loss(out, batch, batch_idx, "mv_loss")
            loss_dict.update(view_loss)

        # StableDiffusion Loss
        sd_loss = self.sd_loss(out, batch)
        loss_dict.update(sd_loss)

        loss = 0.0

        for name, value in loss_dict.items():
            loss_weighted = value * self.C(self.cfg.loss[name])

            # print(
            #     "name {}, w:{:.6f}, loss {:.6f}, loss_w:{:.6f}".format(
            #         name,
            #         self.C(self.cfg.loss[name]),
            #         value.item(),
            #         loss_weighted.item(),
            #     )
            # )

            loss += loss_weighted

        return {"loss": loss}

    def on_fit_start(self) -> None:
        # build mv guidnace
        if self.cfg.albedo_guidance.model_name == "":
            self.albedo_mv_guidance = None
            super().on_fit_start()
        else:
            self.albedo_mv_prompt_processor = threestudio.find(
                self.cfg.albedo_prompt_processor_type
            )(self.cfg.albedo_prompt_processor)
            self.albedo_mv_guidance = threestudio.find(self.cfg.albedo_guidance_type)(
                self.cfg.albedo_guidance
            )
            # fine space
            super().on_fit_start()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        front = batch
        out = self(front)
        
        # if not batch["front"]["view_step"] == self._val_step:
        #     return
        # front = batch["front"]
        # front["height"] = batch["height"]
        # front["width"] = batch["width"]
        # out = self(front)

        self.save_image_grid(
            f"it{self.true_global_step:06d}-{front['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out["albedo"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["camera_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "camera_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        self._val_step += 1

    def test_step(self, batch, batch_idx):
        front = batch["front"]
        front["height"] = batch["height"]
        front["width"] = batch["width"]

        out = self(front)

        self.save_image_grid(
            f"it{self.true_global_step:06d}-test/{front['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if self.cfg.texture
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["camera_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "camera_normal" in out
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["shading_normal"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": out["albedo"][0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step:06d}-test",
            f"it{self.true_global_step:06d}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
