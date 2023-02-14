import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import logging
from models.utils import common, image, io, path_ops, types, visuals, struct
from typing import Dict, Optional, Tuple
import os.path as osp
import cv2
from .LieAlgebra import se3
from models import geometry
import jax
SPLITS = [
    "train",
    "val",
]

DEFAULT_FACTOR: int = 2

DEFAULT_FACTORS = {
    "nerfies/broom": 4,
    "nerfies/curls": 8,
    "nerfies/tail": 4,
    "nerfies/toby-sit": 4,
    "hypernerf/3dprinter": 4,
    "hypernerf/chicken": 4,
    "hypernerf/peel-banana": 4,
}
DEFAULT_FPS = {
    "nerfies/broom": 15,
    "nerfies/curls": 5,
    "nerfies/tail": 15,
    "nerfies/toby-sit": 15,
    "hypernerf/3dprinter": 15,
    "hypernerf/chicken": 15,
    "hypernerf/peel-banana": 15,
}

def _load_scene_info(
    data_dir: types.PathType,
) -> Tuple[np.ndarray, float, float, float]:
    scene_dict = io.load(osp.join(data_dir, "scene.json"))
    center = np.array(scene_dict["center"], dtype=np.float32)
    scale = scene_dict["scale"]
    near = scene_dict["near"]
    far = scene_dict["far"]
    return center, scale, near, far

def _load_extra_info(self) -> None:
    extra_path = osp.join(self.data_dir, "extra.json")
    extra_dict = io.load(extra_path)
    self._factor = extra_dict["factor"]
    self._fps = extra_dict["fps"]
    self._bbox = np.array(extra_dict["bbox"], dtype=np.float32)
    self._lookat = np.array(extra_dict["lookat"], dtype=np.float32)
    self._up = np.array(extra_dict["up"], dtype=np.float32)
    
def _load_metadata_info(
    data_dir: types.PathType,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_dict = io.load(osp.join(data_dir, "dataset.json"))
    _frame_names = np.array(dataset_dict["ids"])

    metadata_dict = io.load(osp.join(data_dir, "metadata.json"))
    time_ids = np.array(
        [metadata_dict[k]["warp_id"] for k in _frame_names], dtype=np.uint32
    )
    camera_ids = np.array(
        [metadata_dict[k]["camera_id"] for k in _frame_names], dtype=np.uint32
    )

    frame_names_map = np.zeros(
        (time_ids.max() + 1, camera_ids.max() + 1), _frame_names.dtype
    )
    for i, (t, c) in enumerate(zip(time_ids, camera_ids)):
        frame_names_map[t, c] = _frame_names[i]

    return frame_names_map, time_ids, camera_ids


class iPhoneDatasetFromAllFrames():
    def __init__(self, conf, split='train'):
        self.parser = iPhoneParser(conf)
        self.batch_size =  0.28

        self.split = split
        training = None
        if training is None:
            training = self.split is not None and self.split.startswith(
                "train"
            )
        
        self.training = training
 
        self.bkgd_points_batch_size = 1 - self.batch_size
        (
            self._frame_names,
            self._time_ids,
            self._camera_ids,
        ) = self.parser.load_split(self.split)

        # Preload cameras and points since it has small memory footprint.
        self.cameras = common.parallel_map(
            self.parser.load_camera, self._time_ids, self._camera_ids
        )

        if self.training:
            # If training, we need to make sure the unique metadata are
            # consecutive.
            self.validate_metadata_info()
            
        if self.training:
            rgbas = np.array(
                    common.parallel_map(
                        self.parser.load_rgba, self._time_ids, self._camera_ids
                    )
            ).reshape(-1, 4)
            
            self.rgbs, self.masks = rgbas[..., :3], rgbas[..., -1:]
            
            rays = jax.tree_map(
                lambda x: x.reshape(-1, x.shape[-1]),
                [
                    c.pixels_to_rays(c.get_pixels())._replace(
                        metadata=struct.Metadata(
                            time=np.full(
                                tuple(c.image_shape) + (1,),
                                ti,
                                dtype=np.uint32,
                            ),
                            camera=np.full(
                                tuple(c.image_shape) + (1,),
                                ci,
                                dtype=np.uint32,
                            ),
                        )
                    )
                    for c, ti, ci in zip(
                        self.cameras, self._time_ids, self._camera_ids
                    )
                ],
            )
            self.rays = struct.Rays(
                origins=np.concatenate([r.origins for r in rays], axis=0),
                directions=np.concatenate(
                    [r.directions for r in rays], axis=0
                ),
                metadata=struct.Metadata(
                    time=np.concatenate(
                        [r.metadata.time for r in rays], axis=0
                    ),
                    camera=np.concatenate(
                        [r.metadata.camera for r in rays], axis=0
                    ),
                ),
            ) 
            self.depths = np.array(
                common.parallel_map(
                    self.parser.load_depth,
                    self._time_ids,
                    self._camera_ids,
                )
            ).reshape(-1, 1)
        import ipdb; ipdb.set_trace()
        # if self.bkgd_points_batch_size > 0:
        #     self.bkgd_points = self.parser.load_bkgd_points()
        #     # Weird batching logic from the original HyperNeRF repo.
        #     self.bkgd_points_batch_size = min(
        #         self.bkgd_points_batch_size,
        #         len(self.bkgd_points),
        #     )
        #     self.bkgd_points_batch_size -= self.bkgd_points_batch_size
    def gen_rays_at_depth(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # Normalize
        npixels_x = (pixels_x - (self.W-1)/2) / ((self.W-1)/2)
        npixels_y = (pixels_y - (self.H-1)/2) / ((self.H-1)/2)
        mask = self.masks[img_idx].permute(2,0,1)[None,...].to(self.device) # 1, 3, H, W
        mask = mask[:,:1,...] # 1, 1, H, W
        depth = self.depths[img_idx][None,None,...].to(self.device) # 1, 1, H, W
        npixels = torch.cat([npixels_x.unsqueeze(-1).unsqueeze(0),npixels_y.unsqueeze(-1).unsqueeze(0)], dim=-1) # 1, W, H, 2
        # grid_sample: sample image on (x_i, y_i)
        mask = F.grid_sample(mask, npixels, mode='nearest', padding_mode='border', align_corners=True).squeeze()[...,None] # W, H, 1
        depth = F.grid_sample(depth, npixels, padding_mode='border', align_corners=True).squeeze()[...,None] # W, H, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p_d = p.clone().detach()
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, None, :3, :3], p_d[:, :, :, None]).squeeze() * self.scales_all[img_idx, :] # W, H, 3
        rays_s = torch.matmul(pose[None, None, :3, :3], p_d[:, :, :, None]).squeeze() # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0,1), rays_v.transpose(0,1), rays_s.transpose(0,1), mask.transpose(0,1)


    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        mask = self.masks[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1) # batch_size, 10

    def gen_random_rays_at_depth_paint(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        mask = self.masks[img_idx].to(self.device)
        coord_mask = np.argwhere((mask >0.9).detach().cpu().numpy())
        coord_mask = coord_mask[np.random.randint(0, len(coord_mask),
                                                      batch_size)]
        pixels_x = torch.from_numpy(coord_mask[..., 1]).to(self.device)
        pixels_y = torch.from_numpy(coord_mask[..., 0]).to(self.device)
        # pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        # pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)
        
        depth = self.depths[img_idx].to(self.device)
        color = color[(pixels_y, pixels_x)] # batch_size, 3
        mask = mask[(pixels_y, pixels_x)] # batch_size, 3
        depth = depth[(pixels_y, pixels_x)][..., None] # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        p_d = p.clone().detach()
        # pixel -> camera -> normalization space (w/o pose). ps: 'pose' is a gap between camera and world
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, :3, :3], p_d[:, :, None]).squeeze() * self.scales_all[img_idx, :] # batch_size, 3
        rays_l = torch.linalg.norm(p_d, ord=2, dim=-1, keepdim=True) # batch_size, 1
        rays_s = torch.matmul(pose[None, :3, :3], p_d[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, rays_s, rays_l, color, mask[:, :1]], dim=-1) # batch_size, 14

    # Depth
    def gen_random_rays_at_depth(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)
        mask = self.masks[img_idx].to(self.device)
        depth = self.depths[img_idx].to(self.device)
        color = color[(pixels_y, pixels_x)] # batch_size, 3
        mask = mask[(pixels_y, pixels_x)] # batch_size, 3
        depth = depth[(pixels_y, pixels_x)][..., None] # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        p_d = p.clone().detach()
        # pixel -> camera -> normalization space (w/o pose). ps: 'pose' is a gap between camera and world
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, :3, :3], p_d[:, :, None]).squeeze() * self.scales_all[img_idx, :] # batch_size, 3
        rays_l = torch.linalg.norm(p_d, ord=2, dim=-1, keepdim=True) # batch_size, 1
        rays_s = torch.matmul(pose[None, :3, :3], p_d[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, rays_s, rays_l, color, mask[:, :1]], dim=-1) # batch_size, 14

            

    @property
    def has_novel_view(self):
        return (
            len(io.load(osp.join(self.data_dir, "dataset.json"))["val_ids"])
            > 0
        )


class iPhoneParser:
    def __init__(self, conf):
        super(iPhoneParser, self).__init__()
        logging.info('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.dtype = torch.get_default_dtype()

        # Camera
        # self.is_monocular = conf.get_bool('is_monocular')
        self.camera_trainable = conf.get_bool('camera_trainable')

        self.data_dir = conf.get_string('data_dir')
        self.use_undistort = False
        (
            self._center,
            self._scale,
            self._near,
            self._far,
        ) = _load_scene_info(self.data_dir)
        
        (
            self._frame_names_map,
            self._time_ids,
            self._camera_ids,
        ) = _load_metadata_info(self.data_dir)
        self._load_extra_info()
        self.splits_dir = osp.join(self.data_dir, "splits")
        if not osp.exists(self.splits_dir):
            self._create_splits()
            
        def load_rgba_up(
            self,
            time_id: int,
            camera_id: int,
            *,
            use_undistort: Optional[bool] = None,
        ) -> np.ndarray:
            if use_undistort is None:
                use_undistort = self.use_undistort

            frame_name = self._frame_names_map[time_id, camera_id]
            rgb_path = osp.join(
                self.data_dir,
                "rgb" if not use_undistort else "rgb_undistort",
                f"{self._factor}x",
                frame_name + ".png",
            )
            if osp.exists(rgb_path):
                rgba = io.load(rgb_path, flags=cv2.IMREAD_UNCHANGED)
                if rgba.shape[-1] == 3:
                    rgba = np.concatenate(
                        [rgba, np.full_like(rgba[..., :1], 255)], axis=-1
                    )
            elif use_undistort:
                camera = self.load_camera(time_id, camera_id, use_undistort=False)
                rgb = self.load_rgba(time_id, camera_id, use_undistort=False)[
                    ..., :3
                ]
                rgb = cv2.undistort(rgb, camera.intrin, camera.distortion)
                alpha = (
                    cv2.undistort(
                        np.full_like(rgb, 255),
                        camera.intrin,
                        camera.distortion,
                    )
                    == 255
                )[..., :1].astype(np.uint8) * 255
                rgba = np.concatenate([rgb, alpha], axis=-1)
                io.dump(rgb_path, rgba)
            else:
                raise ValueError(f"RGB image not found: {rgb_path}.")
            return rgba
    
        def load_rgba(self, time_id: int, camera_id: int) -> np.ndarray:
            
            return self.load_rgba_up(time_id, camera_id, use_undistort=False)

        def load_depth(
            self,
            time_id: int,
            camera_id: int,
        ) -> np.ndarray:
            frame_name = self._frame_names_map[time_id, camera_id]
            depth_path = osp.join(
                self.data_dir, "depth", f"{self._factor}x", frame_name + ".npy"
            )
            depth = io.load(depth_path) * self.scale
            camera = self.load_camera(time_id, camera_id)
            # The original depth data is projective; convert it to ray traveling
            # distance.
            depth = depth / camera.pixels_to_cosa(camera.get_pixels())
            return depth

        def load_camera(
            self, time_id: int, camera_id: int, **_
        ) -> geometry.Camera:
            return super().load_camera(time_id, camera_id, use_undistort=False)

        def load_covisible(
            self, time_id: int, camera_id: int, split: str, **_
        ) -> np.ndarray:
            return super().load_covisible(
                time_id, camera_id, split, use_undistort=False
            )

        def load_keypoints(
            self, time_id: int, camera_id: int, split: str, **_
        ) -> np.ndarray:
            return super().load_keypoints(
                time_id, camera_id, split, use_undistort=False
            )

        def _load_extra_info(self) -> None:
            extra_path = osp.join(self.data_dir, "extra.json")
            extra_dict = io.load(extra_path)
            self._factor = extra_dict["factor"]
            self._fps = extra_dict["fps"]
            self._bbox = np.array(extra_dict["bbox"], dtype=np.float32)
            self._lookat = np.array(extra_dict["lookat"], dtype=np.float32)
            self._up = np.array(extra_dict["up"], dtype=np.float32)

        def _create_splits(self):
            def _create_split(split):
                assert split in self.SPLITS, f'Unknown split "{split}".'

                if split == "train":
                    mask = self.camera_ids == 0
                elif split == "val":
                    mask = self.camera_ids != 0
                else:
                    raise ValueError(f"Unknown split {split}.")

                frame_names = self.frame_names[mask]
                time_ids = self.time_ids[mask]
                camera_ids = self.camera_ids[mask]
                split_dict = {
                    "frame_names": frame_names,
                    "time_ids": time_ids,
                    "camera_ids": camera_ids,
                }
                io.dump(osp.join(self.splits_dir, f"{split}.json"), split_dict)

            common.parallel_map(_create_split, self.SPLITS)


        # self.render_cameras_name = conf.get_string('render_cameras_name')
        # self.object_cameras_name = conf.get_string('object_cameras_name')

        # self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        # self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        # camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        # self.camera_dict = camera_dict
        # self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.jpg')))
        # if len(self.images_lis) == 0:
        #     self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.png')))
        # self.n_images = len(self.images_lis)
        # self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        # self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
        # if len(self.masks_lis) == 0:
        #     self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        # self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # # Depth
        # self.use_depth = conf.get_bool('use_depth')
        # if self.use_depth:
        #     self.depth_scale = conf.get_float('depth_scale', default=1000.)
        #     self.depths_lis = sorted(glob(os.path.join(self.data_dir, 'depth/*.jpg')))
        #     if len(self.depths_lis) == 0:
        #         self.depths_lis = sorted(glob(os.path.join(self.data_dir, 'depth/*.png')))
        #     self.depths_np = np.stack([cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in self.depths_lis]) / self.depth_scale
        #     self.depths_np[self.depths_np == 0] = -1. # avoid nan values
        #     self.depths = torch.from_numpy(self.depths_np.astype(np.float32)).to(self.dtype).cpu()

        # # world_mat is a projection matrix from world to image
        # self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # self.scale_mats_np = []

        # # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        # self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # intrinsics_all = []
        # poses_all = []
        # # Depth, needs x,y,z have equal scale
        # self.scales_all = []

        # for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
        #     P = world_mat @ scale_mat
        #     P = P[:3, :4]
        #     intrinsics, pose = load_K_Rt_from_P(None, P)
        #     # Depth
        #     self.scales_all.append(np.array([1.0/scale_mat[0,0]]))
        #     intrinsics_all.append(torch.from_numpy(intrinsics).to(self.dtype))
        #     poses_all.append(torch.from_numpy(pose).to(self.dtype)) # the inverse of extrinsic matrix

        # self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 3]
        # self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 3]
        # # Depth
        # self.scales_all = torch.from_numpy(np.stack(self.scales_all)).to(self.dtype).to(self.device) # [n_images, 3]
        # intrinsics_all = torch.stack(intrinsics_all).to(self.device) # [n_images, 4, 4]
        # poses_all = torch.stack(poses_all).to(self.device) # [n_images, 4, 4]
        # self.H, self.W = self.images.shape[1], self.images.shape[2]

        # # Camera
        # if self.is_monocular:
        #     self.intrinsics_paras = torch.stack((intrinsics_all[:1,0,0], intrinsics_all[:1,1,1], \
        #                                         intrinsics_all[:1,0,2], intrinsics_all[:1,1,2]),
        #                                             dim=1) # [1, 4]: (fx, fy, cx, cy)
        # else:
        #     self.intrinsics_paras = torch.stack((intrinsics_all[:,0,0], intrinsics_all[:,1,1], \
        #                                         intrinsics_all[:,0,2], intrinsics_all[:,1,2]),
        #                                             dim=1) # [n_images, 4]: (fx, fy, cx, cy)
        # # Depth
        # if self.use_depth:
        #     self.depth_intrinsics_paras = self.intrinsics_paras.clone().detach()
        # self.poses_paras = se3.log(poses_all) # [n_images, 6]
        # if self.camera_trainable:
        #     self.intrinsics_paras.requires_grad_()
        #     if self.use_depth:
        #         self.depth_intrinsics_paras.requires_grad_()
        #     self.poses_paras.requires_grad_()
        # else:
        #     self.static_paras_to_mat()

        # object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        # object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # # Object scale mat: region of interest to **extract mesh**
        # object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        # object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        # object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        # self.object_bbox_min = object_bbox_min[:3, 0]
        # self.object_bbox_max = object_bbox_max[:3, 0]

        logging.info('Load data: End')


    # Camera
    def static_paras_to_mat(self):
        fx, fy, cx, cy = self.intrinsics_paras[:,0], self.intrinsics_paras[:,1],\
                            self.intrinsics_paras[:,2], self.intrinsics_paras[:,3]
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsics_all_inv_mat = torch.stack((torch.stack(
                                (1/fx, zeros, -cx/fx), dim=1), torch.stack(
                                (zeros, 1/fy, -cy/fy), dim=1), torch.stack(
                                (zeros, zeros, ones), dim=1)),
                                    dim=1)
        self.intrinsics_all_inv = torch.cat((torch.cat(
                                (intrinsics_all_inv_mat, torch.stack(
                                (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                    dim=1)
        if self.use_depth:
            fx_d, fy_d, cx_d, cy_d = self.depth_intrinsics_paras[:,0], self.depth_intrinsics_paras[:,1],\
                            self.depth_intrinsics_paras[:,2], self.depth_intrinsics_paras[:,3]
            zeros = torch.zeros_like(fx_d)
            ones = torch.ones_like(fx_d)
            depth_intrinsics_all_inv_mat = torch.stack((torch.stack(
                                    (1/fx_d, zeros, -cx_d/fx_d), dim=1), torch.stack(
                                    (zeros, 1/fy_d, -cy_d/fy_d), dim=1), torch.stack(
                                    (zeros, zeros, ones), dim=1)),
                                        dim=1)
            self.depth_intrinsics_all_inv = torch.cat((torch.cat(
                                    (depth_intrinsics_all_inv_mat, torch.stack(
                                    (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                    (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                        dim=1)
        self.poses_all = se3.exp(self.poses_paras)


    def dynamic_paras_to_mat(self, img_idx, add_depth=False):
        if self.is_monocular:
            intrinsic_paras = self.intrinsics_paras[:1, :]
        else:
            intrinsic_paras = self.intrinsics_paras[img_idx:(img_idx+1), :]
        fx, fy, cx, cy = intrinsic_paras[:,0], intrinsic_paras[:,1], intrinsic_paras[:,2], intrinsic_paras[:,3]
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsics_inv_mat = torch.stack((torch.stack(
                                (1/fx, zeros, -cx/fx), dim=1), torch.stack(
                                (zeros, 1/fy, -cy/fy), dim=1), torch.stack(
                                (zeros, zeros, ones), dim=1)),
                                    dim=1)
        intrinsic_inv = torch.cat((torch.cat(
                                (intrinsics_inv_mat, torch.stack(
                                (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                    dim=1)
        pose_paras = self.poses_paras[img_idx:(img_idx+1), :]
        pose = se3.exp(pose_paras)
        if add_depth:
            if self.is_monocular:
                depth_intrinsic_paras = self.depth_intrinsics_paras[:1, :]
            else:
                depth_intrinsic_paras = self.depth_intrinsics_paras[img_idx:(img_idx+1), :]
            fx_d, fy_d, cx_d, cy_d = depth_intrinsic_paras[:,0], depth_intrinsic_paras[:,1],\
                                        depth_intrinsic_paras[:,2], depth_intrinsic_paras[:,3]
            zeros = torch.zeros_like(fx_d)
            ones = torch.ones_like(fx_d)
            depth_intrinsics_inv_mat = torch.stack((torch.stack(
                                    (1/fx_d, zeros, -cx_d/fx_d), dim=1), torch.stack(
                                    (zeros, 1/fy_d, -cy_d/fy_d), dim=1), torch.stack(
                                    (zeros, zeros, ones), dim=1)),
                                        dim=1)
            depth_intrinsic_inv = torch.cat((torch.cat(
                                    (depth_intrinsics_inv_mat, torch.stack(
                                    (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                    (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                        dim=1)
            return intrinsic_inv.squeeze(), pose.squeeze(), depth_intrinsic_inv.squeeze()
        return intrinsic_inv.squeeze(), pose.squeeze()


    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    # Depth
    def gen_rays_at_depth(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # Normalize
        npixels_x = (pixels_x - (self.W-1)/2) / ((self.W-1)/2)
        npixels_y = (pixels_y - (self.H-1)/2) / ((self.H-1)/2)
        mask = self.masks[img_idx].permute(2,0,1)[None,...].to(self.device) # 1, 3, H, W
        mask = mask[:,:1,...] # 1, 1, H, W
        depth = self.depths[img_idx][None,None,...].to(self.device) # 1, 1, H, W
        npixels = torch.cat([npixels_x.unsqueeze(-1).unsqueeze(0),npixels_y.unsqueeze(-1).unsqueeze(0)], dim=-1) # 1, W, H, 2
        # grid_sample: sample image on (x_i, y_i)
        mask = F.grid_sample(mask, npixels, mode='nearest', padding_mode='border', align_corners=True).squeeze()[...,None] # W, H, 1
        depth = F.grid_sample(depth, npixels, padding_mode='border', align_corners=True).squeeze()[...,None] # W, H, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p_d = p.clone().detach()
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, None, :3, :3], p_d[:, :, :, None]).squeeze() * self.scales_all[img_idx, :] # W, H, 3
        rays_s = torch.matmul(pose[None, None, :3, :3], p_d[:, :, :, None]).squeeze() # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0,1), rays_v.transpose(0,1), rays_s.transpose(0,1), mask.transpose(0,1)


    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        mask = self.masks[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1) # batch_size, 10

    def gen_random_rays_at_depth_paint(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        mask = self.masks[img_idx].to(self.device)
        coord_mask = np.argwhere((mask >0.9).detach().cpu().numpy())
        coord_mask = coord_mask[np.random.randint(0, len(coord_mask),
                                                      batch_size)]
        pixels_x = torch.from_numpy(coord_mask[..., 1]).to(self.device)
        pixels_y = torch.from_numpy(coord_mask[..., 0]).to(self.device)
        # pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        # pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)
        
        depth = self.depths[img_idx].to(self.device)
        color = color[(pixels_y, pixels_x)] # batch_size, 3
        mask = mask[(pixels_y, pixels_x)] # batch_size, 3
        depth = depth[(pixels_y, pixels_x)][..., None] # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        p_d = p.clone().detach()
        # pixel -> camera -> normalization space (w/o pose). ps: 'pose' is a gap between camera and world
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, :3, :3], p_d[:, :, None]).squeeze() * self.scales_all[img_idx, :] # batch_size, 3
        rays_l = torch.linalg.norm(p_d, ord=2, dim=-1, keepdim=True) # batch_size, 1
        rays_s = torch.matmul(pose[None, :3, :3], p_d[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, rays_s, rays_l, color, mask[:, :1]], dim=-1) # batch_size, 14

    # Depth
    def gen_random_rays_at_depth(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)
        mask = self.masks[img_idx].to(self.device)
        depth = self.depths[img_idx].to(self.device)
        color = color[(pixels_y, pixels_x)] # batch_size, 3
        mask = mask[(pixels_y, pixels_x)] # batch_size, 3
        depth = depth[(pixels_y, pixels_x)][..., None] # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        p_d = p.clone().detach()
        # pixel -> camera -> normalization space (w/o pose). ps: 'pose' is a gap between camera and world
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, :3, :3], p_d[:, :, None]).squeeze() * self.scales_all[img_idx, :] # batch_size, 3
        rays_l = torch.linalg.norm(p_d, ord=2, dim=-1, keepdim=True) # batch_size, 1
        rays_s = torch.matmul(pose[None, :3, :3], p_d[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, rays_s, rays_l, color, mask[:, :1]], dim=-1) # batch_size, 14


    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far


    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


    def get_image_size(self):
        return self.H, self.W

    def depth_at(self, idx, resolution_level):
        depth_img = cv.resize(self.depths_np[idx],
                                (self.W // resolution_level, self.H // resolution_level))
        depth_img = 255. - np.clip(depth_img/depth_img.max(), a_max=1, a_min=0) * 255.
        return cv.applyColorMap(np.uint8(depth_img), cv.COLORMAP_JET)


def poses_avg(poses):
    # center = poses[:, :3, 3].mean(0)
    # forward = poses[:, :3, 2].sum(0)
    # up = poses[:, :3, 1].sum(0)
    # c2w = view_matrix(forward, up, center)

    #######FOR GIRL########
    center = poses[0, :3, 3]
    forward = poses[0, :3, 2]
    up = poses[0, :3, 1]
    c2w = view_matrix(forward, up, center)
    return c2w

def view_matrix(
        forward: np.ndarray,
        up: np.ndarray,
        cam_location: np.ndarray):
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    mat = np.stack((rot_x, rot_y, rot_z, cam_location), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def normalize(vec, axis=-1):
    return vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + 1e-9)


def c2w_track_spiral(c2w, up_vec, rads, focus: float, zrate: float, rots: int, N: int, zdelta: float = 0.):
    # TODO: support zdelta
    """generate camera to world matrices of spiral track, looking at the same point [0,0,focus]
    Args:
        c2w ([4,4] or [3,4]):   camera to world matrix (of the spiral center, with average rotation and average translation)
        up_vec ([3,]):          vector pointing up
        rads ([3,]):            radius of x,y,z direction, of the spiral track
        # zdelta ([float]):       total delta z that is allowed to change 
        focus (float):          a focus value (to be looked at) (in camera coordinates)
        zrate ([float]):        a factor multiplied to z's angle
        rots ([int]):           number of rounds to rotate
        N ([int]):              number of total views
    """

    c2w_tracks = []
    rads = np.array(list(rads) + [1.])

    # focus_in_cam = np.array([0, 0, -focus, 1.])   # openGL convention
    focus_in_cam = np.array([0, 0, focus, 1.])      # openCV convention
    focus_in_world = np.dot(c2w[:3, :4], focus_in_cam)
    rots = 1
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        cam_location = np.dot(
            c2w[:3, :4],
            # np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads    # openGL convention
            np.array([np.cos(theta), np.sin(theta), np.sin(
                theta*zrate), 1.]) * rads        # openCV convention
        )

    center = c2w[:3, 3].reshape(3)
    print("center", center)
    rad = 0.8
    for theta in np.linspace(0, 2 * np.pi, N+1)[:-1]:
        cam_location = np.zeros(3)
        x = center[0] + rad*np.cos(theta)
        y = center[1] + rad*np.sin(theta)
        z = center[2]
        cam_location[0] = x
        cam_location[1] = y
        cam_location[2] = z

        c2w_i = look_at(cam_location, focus_in_world, up=up_vec)
        c2w_tracks.append(c2w_i)
    return c2w_tracks


def look_at(
    cam_location: np.ndarray,
    point: np.ndarray,
    up=np.array([0., -1., 0.])          # openCV convention
    # up=np.array([0., 1., 0.])         # openGL convention
):
    # Cam points in positive z direction
    forward = normalize(point - cam_location)     # openCV convention
    # forward = normalize(cam_location - point)   # openGL convention
    return view_matrix(forward, up, cam_location)

class Dataset_nv:
    def __init__(self, conf):
        super(Dataset_nv, self).__init__()
        logging.info('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.dtype = torch.get_default_dtype()

        # Camera
        self.is_monocular = conf.get_bool('is_monocular')
        self.camera_trainable = conf.get_bool('camera_trainable')

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.jpg')))
        if len(self.images_lis) == 0:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'rgb/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
        if len(self.masks_lis) == 0:
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # Depth
        self.use_depth = conf.get_bool('use_depth')
        if self.use_depth:
            self.depth_scale = conf.get_float('depth_scale', default=1000.)
            self.depths_lis = sorted(glob(os.path.join(self.data_dir, 'depth/*.jpg')))
            if len(self.depths_lis) == 0:
                self.depths_lis = sorted(glob(os.path.join(self.data_dir, 'depth/*.png')))
            self.depths_np = np.stack([cv.imread(im_name, cv.IMREAD_UNCHANGED) for im_name in self.depths_lis]) / self.depth_scale
            self.depths_np[self.depths_np == 0] = -1. # avoid nan values
            self.depths = torch.from_numpy(self.depths_np.astype(np.float32)).to(self.dtype).cpu()

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        intrinsics_all = []
        poses_all = []
        # Depth, needs x,y,z have equal scale
        self.scales_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            # Depth
            self.scales_all.append(np.array([1.0/scale_mat[0,0]]))
            intrinsics_all.append(torch.from_numpy(intrinsics).to(self.dtype))
            poses_all.append(torch.from_numpy(pose).to(self.dtype)) # the inverse of extrinsic matrix
     
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.dtype).cpu() # [n_images, H, W, 3]
        # Depth
        self.scales_all = torch.from_numpy(np.stack(self.scales_all)).to(self.dtype).to(self.device) # [n_images, 3]
        intrinsics_all = torch.stack(intrinsics_all).to(self.device) # [n_images, 4, 4]
        poses_all = torch.stack(poses_all) # [n_images, 4, 4]
        c2w_center = poses_avg(poses_all.cpu().numpy())
        up = poses_all[:, :3, 1].sum(0)
        rads = np.percentile(np.abs(poses_all[:, :3, 3]), 85, 0)
        focus_distance = np.mean(np.linalg.norm(poses_all[:, :3, 3], axis=-1))
        poses_all = c2w_track_spiral(
            c2w_center, up, rads, focus_distance*0.8, zrate=0.0, rots=1, N=len(poses_all))
        poses_all = torch.from_numpy(np.stack(poses_all)).to(self.device).float()
        self.H, self.W = self.images.shape[1], self.images.shape[2]

        # Camera
        if self.is_monocular:
            self.intrinsics_paras = torch.stack((intrinsics_all[:1,0,0], intrinsics_all[:1,1,1], \
                                                intrinsics_all[:1,0,2], intrinsics_all[:1,1,2]),
                                                    dim=1) # [1, 4]: (fx, fy, cx, cy)
        else:
            self.intrinsics_paras = torch.stack((intrinsics_all[:,0,0], intrinsics_all[:,1,1], \
                                                intrinsics_all[:,0,2], intrinsics_all[:,1,2]),
                                                    dim=1) # [n_images, 4]: (fx, fy, cx, cy)
        # Depth
        if self.use_depth:
            self.depth_intrinsics_paras = self.intrinsics_paras.clone().detach()
        self.poses_paras = se3.log(poses_all) # [n_images, 6]
        if self.camera_trainable:
            self.intrinsics_paras.requires_grad_()
            if self.use_depth:
                self.depth_intrinsics_paras.requires_grad_()
            self.poses_paras.requires_grad_()
        else:
            self.static_paras_to_mat()

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        logging.info('Load data: End')


    # Camera
    def static_paras_to_mat(self):
        fx, fy, cx, cy = self.intrinsics_paras[:,0], self.intrinsics_paras[:,1],\
                            self.intrinsics_paras[:,2], self.intrinsics_paras[:,3]
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsics_all_inv_mat = torch.stack((torch.stack(
                                (1/fx, zeros, -cx/fx), dim=1), torch.stack(
                                (zeros, 1/fy, -cy/fy), dim=1), torch.stack(
                                (zeros, zeros, ones), dim=1)),
                                    dim=1)
        self.intrinsics_all_inv = torch.cat((torch.cat(
                                (intrinsics_all_inv_mat, torch.stack(
                                (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                    dim=1)
        if self.use_depth:
            fx_d, fy_d, cx_d, cy_d = self.depth_intrinsics_paras[:,0], self.depth_intrinsics_paras[:,1],\
                            self.depth_intrinsics_paras[:,2], self.depth_intrinsics_paras[:,3]
            zeros = torch.zeros_like(fx_d)
            ones = torch.ones_like(fx_d)
            depth_intrinsics_all_inv_mat = torch.stack((torch.stack(
                                    (1/fx_d, zeros, -cx_d/fx_d), dim=1), torch.stack(
                                    (zeros, 1/fy_d, -cy_d/fy_d), dim=1), torch.stack(
                                    (zeros, zeros, ones), dim=1)),
                                        dim=1)
            self.depth_intrinsics_all_inv = torch.cat((torch.cat(
                                    (depth_intrinsics_all_inv_mat, torch.stack(
                                    (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                    (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                        dim=1)
        self.poses_all = se3.exp(self.poses_paras)


    def dynamic_paras_to_mat(self, img_idx, add_depth=False):
        if self.is_monocular:
            intrinsic_paras = self.intrinsics_paras[:1, :]
        else:
            intrinsic_paras = self.intrinsics_paras[img_idx:(img_idx+1), :]
        fx, fy, cx, cy = intrinsic_paras[:,0], intrinsic_paras[:,1], intrinsic_paras[:,2], intrinsic_paras[:,3]
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsics_inv_mat = torch.stack((torch.stack(
                                (1/fx, zeros, -cx/fx), dim=1), torch.stack(
                                (zeros, 1/fy, -cy/fy), dim=1), torch.stack(
                                (zeros, zeros, ones), dim=1)),
                                    dim=1)
        intrinsic_inv = torch.cat((torch.cat(
                                (intrinsics_inv_mat, torch.stack(
                                (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                    dim=1)
        pose_paras = self.poses_paras[img_idx:(img_idx+1), :]
        pose = se3.exp(pose_paras)
        if add_depth:
            if self.is_monocular:
                depth_intrinsic_paras = self.depth_intrinsics_paras[:1, :]
            else:
                depth_intrinsic_paras = self.depth_intrinsics_paras[img_idx:(img_idx+1), :]
            fx_d, fy_d, cx_d, cy_d = depth_intrinsic_paras[:,0], depth_intrinsic_paras[:,1],\
                                        depth_intrinsic_paras[:,2], depth_intrinsic_paras[:,3]
            zeros = torch.zeros_like(fx_d)
            ones = torch.ones_like(fx_d)
            depth_intrinsics_inv_mat = torch.stack((torch.stack(
                                    (1/fx_d, zeros, -cx_d/fx_d), dim=1), torch.stack(
                                    (zeros, 1/fy_d, -cy_d/fy_d), dim=1), torch.stack(
                                    (zeros, zeros, ones), dim=1)),
                                        dim=1)
            depth_intrinsic_inv = torch.cat((torch.cat(
                                    (depth_intrinsics_inv_mat, torch.stack(
                                    (zeros, zeros, zeros), dim=1)[...,None]), dim=-1), torch.stack(
                                    (zeros, zeros, zeros, ones), dim=1)[:,None,:]),
                                        dim=1)
            return intrinsic_inv.squeeze(), pose.squeeze(), depth_intrinsic_inv.squeeze()
        return intrinsic_inv.squeeze(), pose.squeeze()


    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    # Depth
    def gen_rays_at_depth(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # Normalize
        npixels_x = (pixels_x - (self.W-1)/2) / ((self.W-1)/2)
        npixels_y = (pixels_y - (self.H-1)/2) / ((self.H-1)/2)
        mask = self.masks[img_idx].permute(2,0,1)[None,...].to(self.device) # 1, 3, H, W
        mask = mask[:,:1,...] # 1, 1, H, W
        depth = self.depths[img_idx][None,None,...].to(self.device) # 1, 1, H, W
        npixels = torch.cat([npixels_x.unsqueeze(-1).unsqueeze(0),npixels_y.unsqueeze(-1).unsqueeze(0)], dim=-1) # 1, W, H, 2
        # grid_sample: sample image on (x_i, y_i)
        mask = F.grid_sample(mask, npixels, mode='nearest', padding_mode='border', align_corners=True).squeeze()[...,None] # W, H, 1
        depth = F.grid_sample(depth, npixels, padding_mode='border', align_corners=True).squeeze()[...,None] # W, H, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p_d = p.clone().detach()
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, None, :3, :3], p_d[:, :, :, None]).squeeze() * self.scales_all[img_idx, :] # W, H, 3
        rays_s = torch.matmul(pose[None, None, :3, :3], p_d[:, :, :, None]).squeeze() # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0,1), rays_v.transpose(0,1), rays_s.transpose(0,1), mask.transpose(0,1)


    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        mask = self.masks[img_idx].to(self.device)[(pixels_y, pixels_x)] # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose = self.dynamic_paras_to_mat(img_idx)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1) # batch_size, 10

    def gen_random_rays_at_depth_paint(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        mask = self.masks[img_idx].to(self.device)
        coord_mask = np.argwhere((mask >0.9).detach().cpu().numpy())
        coord_mask = coord_mask[np.random.randint(0, len(coord_mask),
                                                      batch_size)]
        pixels_x = torch.from_numpy(coord_mask[..., 1]).to(self.device)
        pixels_y = torch.from_numpy(coord_mask[..., 0]).to(self.device)
        # pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        # pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)
        
        depth = self.depths[img_idx].to(self.device)
        color = color[(pixels_y, pixels_x)] # batch_size, 3
        mask = mask[(pixels_y, pixels_x)] # batch_size, 3
        depth = depth[(pixels_y, pixels_x)][..., None] # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        p_d = p.clone().detach()
        # pixel -> camera -> normalization space (w/o pose). ps: 'pose' is a gap between camera and world
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, :3, :3], p_d[:, :, None]).squeeze() * self.scales_all[img_idx, :] # batch_size, 3
        rays_l = torch.linalg.norm(p_d, ord=2, dim=-1, keepdim=True) # batch_size, 1
        rays_s = torch.matmul(pose[None, :3, :3], p_d[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, rays_s, rays_l, color, mask[:, :1]], dim=-1) # batch_size, 14

    # Depth
    def gen_random_rays_at_depth(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx].to(self.device)
        mask = self.masks[img_idx].to(self.device)
        depth = self.depths[img_idx].to(self.device)
        color = color[(pixels_y, pixels_x)] # batch_size, 3
        mask = mask[(pixels_y, pixels_x)] # batch_size, 3
        depth = depth[(pixels_y, pixels_x)][..., None] # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.dtype) # batch_size, 3
        p_d = p.clone().detach()
        # pixel -> camera -> normalization space (w/o pose). ps: 'pose' is a gap between camera and world
        # Camera
        if self.camera_trainable:
            intrinsic_inv, pose, depth_intrinsic_inv = self.dynamic_paras_to_mat(img_idx, add_depth=True)
        else:
            if self.is_monocular:
                intrinsic_inv = self.intrinsics_all_inv[0]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[0]
            else:
                intrinsic_inv = self.intrinsics_all_inv[img_idx]
                depth_intrinsic_inv = self.depth_intrinsics_all_inv[img_idx]
            pose = self.poses_all[img_idx]
        p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(pose[None, :3, :3], rays_v[:, :, None]).squeeze() # batch_size, 3
        p_d = depth * torch.matmul(depth_intrinsic_inv[None, :3, :3], p_d[:, :, None]).squeeze() * self.scales_all[img_idx, :] # batch_size, 3
        rays_l = torch.linalg.norm(p_d, ord=2, dim=-1, keepdim=True) # batch_size, 1
        rays_s = torch.matmul(pose[None, :3, :3], p_d[:, :, None]).squeeze() # batch_size, 3
        rays_o = pose[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, rays_s, rays_l, color, mask[:, :1]], dim=-1) # batch_size, 14


    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)


    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far


    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


    def get_image_size(self):
        return self.H, self.W

    def depth_at(self, idx, resolution_level):
        depth_img = cv.resize(self.depths_np[idx],
                                (self.W // resolution_level, self.H // resolution_level))
        depth_img = 255. - np.clip(depth_img/depth_img.max(), a_max=1, a_min=0) * 255.
        return cv.applyColorMap(np.uint8(depth_img), cv.COLORMAP_JET)
