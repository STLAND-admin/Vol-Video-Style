#!/usr/bin/env python3
#
# File   : novel_view.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import dataclasses
import itertools
import os.path as osp
from collections import defaultdict
from typing import Dict, Literal, Optional, Sequence, Union
import torch
import numpy as np
import pytorch3d
import models.geometry as geometry
from models.utils import common, io, struct, types
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)


@dataclasses.dataclass
class VideoConfig(object):
    camera_traj: Literal["fixed", "arc", "lemniscate"]
    time_traj: Literal["fixed", "replay"]
    camera_idx: int = 0
    time_idx: int = 0
    camera_traj_params: Dict = dataclasses.field(
        default_factory=lambda: {"num_frames": 60, "degree": 30}
    )
    fps: float = 20

    def __post_init__(self):
        assert not (self.camera_traj == "fixed" and self.time_traj == "fixed")

    def __repr__(self):
        if self.camera_traj == "fixed":
            return "Stabilized-view video"
        elif self.time_traj == "fixed":
            return "Novel-view video"
        else:
            return "Bullet-time video"

    @property
    def short_name(self):
        fps = float(self.fps)
        if self.camera_traj == "fixed":
            return (
                f"stabilized_view@ci={self.camera_idx}-ti={self.time_idx}-"
                f"fps={fps}"
            )
        elif self.time_traj == "fixed":
            cparams_str = "-".join(
                [f"{k}={v}" for k, v in self.camera_traj_params.items()]
            )
            return (
                f"novel_view@ci={self.camera_idx}-ti={self.time_idx}-"
                f"fps={fps}-ctraj={self.camera_traj}-{cparams_str}"
            )
        else:
            cparams_str = "-".join(
                [f"{k}={v}" for k, v in self.camera_traj_params.items()]
            )
            return (
                f"bullet_time@ci={self.camera_idx}-ti={self.time_idx}-"
                f"fps={fps}-ctraj={self.camera_traj}-{cparams_str}"
            )


class Video():
    """Render video from the dynamic NeRF model.

    Note that for all rgb predictions, we use the quantized version for
    computing metrics such that the results are consistent when loading saved
    images afterwards.

    There are three modes for rendering videos:
        (1) Novel-view rendering, when camera_traj != 'fixed' and time_traj ==
            'fixed'.
        (2) Stabilized-view rendering, when camera_traj == 'fixed' and
            time_traj == 'replay'.
        (3) Bullet-time rendering, when camera_traj != 'fixed' and
            time_traj == 'replay'.
    """

    def __init__(
        self,
        split,
        eval_datasets,
    ):
        super().__init__()

        self.split = split
        configs = [
            {'camera_traj': 'lemniscate', 'time_traj': 'fixed'},
        ]
        self.configs = [VideoConfig(**c) for c in configs]
        self.eval_datasets = eval_datasets
        self.W = eval_datasets.W
        self.H = eval_datasets.H
        self.device = 'cuda'
        self.video_datasets = []
        dataset = self.eval_datasets
        for cfg in self.configs:
            traj_fn = {
                "fixed": lambda c, **_: [c],
                "arc": geometry.get_arc_traj,
                "lemniscate": geometry.get_lemniscate_traj,
            }[cfg.camera_traj]
            cameras = traj_fn(
                dataset.parser.load_camera(
                    dataset.time_ids[cfg.time_idx],
                    dataset.camera_ids[cfg.camera_idx],
                    use_undistort = dataset.parser.use_undistort
                ),
                lookat=dataset.parser.lookat,
                up=dataset.parser.up,
                **cfg.camera_traj_params,
            )
            metadatas = [
                struct.Metadata(
                    time=np.full(tuple(cameras[0].image_shape) + (1,), t),
                    camera=np.full(
                        tuple(cameras[0].image_shape) + (1,), c
                    ),
                )
                for t, c in zip(
                    *{
                        "fixed": [
                            [dataset.time_ids[cfg.time_idx]],
                            [dataset.camera_ids[cfg.camera_idx]],
                        ],
                        # Replay training sequence.
                        "replay": [
                            dataset.time_ids.tolist(),
                            [dataset.camera_ids[cfg.camera_idx]]
                            * dataset.num_times,
                        ],
                    }[cfg.time_traj]
                )
            ]
            if str(cfg) == "Stabilized-view video":
                cfg.fps = dataset.fps
            cameras, metadatas = Video.pad_by_fps(
                cameras,
                metadatas,
                dataset_fps=dataset.parser.fps,
                target_fps=cfg.fps,
            )
            self.video_datasets=[cameras, metadatas]
            self.len = len(cameras)
            image_size = torch.tensor((self.W, self.H)).cuda().long()
            self.image_size = image_size
            self.raster_settings = RasterizationSettings(image_size=(image_size[0].item(), image_size[1].item()),
                                                        blur_radius=0.0,
                                                        faces_per_pixel=1)
    @property
    def eligible(self):
        return len(self.configs) > 0

    @staticmethod
    def pad_by_fps(
        cameras: Sequence[geometry.Camera],
        metadatas: Sequence[struct.Metadata],
        dataset_fps: float,
        target_fps: float,
    ):
        T = len(metadatas)
        V = len(cameras)

        num_time_repeats = max(1, int(target_fps) // int(dataset_fps), V // T)
        num_time_skips = (
            max(1, int(dataset_fps) // int(target_fps))
            if len(metadatas) != 1
            else 1
        )
        metadatas = list(
            itertools.chain(*zip(*(metadatas,) * num_time_repeats))
        )[::num_time_skips]

        num_camera_repeats = len(metadatas) // V
        T = num_camera_repeats * V

        cameras = cameras * num_camera_repeats
        metadatas = metadatas[:T]

        return cameras, metadatas

       
        # Force cull cameras when rendering videos.
    def gen_rays_at(self, index, resolution_level=1):
        data = self.video_datasets

        camera, metadata = data[0][index], data[1][index]
        rays = camera.pixels_to_rays(camera.get_pixels())._replace(
                metadata=metadata
            )
        R = torch.from_numpy(camera.extrin[:3,:3]).cuda().float()
        T = torch.from_numpy(camera.extrin[:3,3]).cuda().float()
        K = torch.from_numpy(camera.intrin).cuda().float()
        cameras_pytorch3d = cameras_from_opencv_projection(R[None], T[None], K[None], self.image_size[None])
        mesh_renderer = MeshRasterizer(cameras=cameras_pytorch3d,
                                raster_settings=self.raster_settings)
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # Normalize
        npixels_x = (pixels_x - (self.W-1)/2) / ((self.W-1)/2)
        npixels_y = (pixels_y - (self.H-1)/2) / ((self.H-1)/2)
        npixels = torch.cat([npixels_x.unsqueeze(-1).unsqueeze(0),npixels_y.unsqueeze(-1).unsqueeze(0)], dim=-1) # 1, W, H, 2
        pixels_x = pixels_x.long()
        pixels_y = pixels_y.long()
        # depth = self.parser.load_depth(time_id, camera_id)
        rays_o = torch.from_numpy(rays.origins).to(self.device)[(pixels_x, pixels_y)]
        rays_v = torch.from_numpy(rays.directions).to(self.device)[(pixels_x, pixels_y)]
        time_id = metadata.time
        time_id = torch.Tensor(time_id.astype(np.float32))
        return rays_o.float(), rays_v.float(), time_id   # mesh_renderer
    
    def __getitem__(self, index):
        data = self.video_datasets[index]
        camera, metadata = data
        rays = camera.pixels_to_rays(camera.get_pixels())._replace(
                metadata=metadata
            )
        return rays

                    
                    
