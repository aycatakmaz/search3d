import jax
import jax.numpy as jnp
jax.devices()
import ml_collections

import hydra
from omegaconf import DictConfig
import numpy as np
from search3d.data.load import Camera, InstanceMasks3D, Images, PointCloud, get_number_of_images
from search3d.utils import get_free_gpu #, create_out_folder
from search3d.mask_feature_computation.features_extractor import FeaturesExtractor

import torch
import os
from glob import glob
import pdb

@hydra.main(config_path="../configs", config_name="search3d_multiscan_eval", version_base=None) 
def main(ctx: DictConfig):

    device = "cpu"
    device = get_free_gpu(7000) if torch.cuda.is_available() else device
    print(f"Using device: {device}")
    if not os.path.exists(ctx.output.mask_features_save_dir):
        os.makedirs(ctx.output.mask_features_save_dir, exist_ok=True)

    if len(ctx.data.masks.masks_path)>0:
        masks_paths = [ctx.data.masks.masks_path]
    else:
        masks_paths = sorted(glob(os.path.join(ctx.data.masks.masks_dir, "*"+ctx.data.masks.masks_suffix)))
    
    features_extractor = FeaturesExtractor(ctx=ctx, device=device)

    for masks_path in masks_paths[:]:
        scene_name = masks_path.split('/')[-1][:-len(ctx.data.masks.masks_suffix)]
        scene_dir = os.path.join(ctx.data.scans_path, scene_name)
        poses_path = os.path.join(scene_dir, ctx.data.camera.poses_path)
        point_cloud_path = os.path.join(scene_dir, ctx.data.point_cloud_suffix.format(scene_name))
        intrinsic_path = os.path.join(scene_dir, ctx.data.camera.intrinsic_path)
        images_path = os.path.join(scene_dir, ctx.data.images.images_path)
        depths_path =os.path.join(scene_dir, ctx.data.depths.depths_path)
        
        # 1. Load the masks
        masks = InstanceMasks3D(masks_path) 

        # 2. Load the images
        indices = np.arange(0, get_number_of_images(poses_path, id_zfill=int(ctx.data.id_zfill), poses_ext=ctx.data.camera.poses_ext), step = ctx.search3d.mask_features.frequency)
        images = Images(images_path=images_path, 
                        extension=ctx.data.images.images_ext, 
                        indices=indices,
                        id_zfill=int(ctx.data.id_zfill))

        # 3. Load the pointcloud
        pointcloud = PointCloud(point_cloud_path)

        # 4. Load the camera configurations
        camera = Camera(intrinsic_path=intrinsic_path, 
                        intrinsic_resolution=ctx.data.camera.intrinsic_resolution, 
                        poses_path=poses_path, 
                        depths_path=depths_path, 
                        extension_depth=ctx.data.depths.depths_ext, 
                        depth_scale=ctx.data.depths.depth_scale,
                        poses_extension=ctx.data.camera.poses_ext,
                        id_zfill=int(ctx.data.id_zfill))

        # 5. Set scene for the features_extracor
        features_extractor.set_scene(camera=camera,
                                    images=images, 
                                    masks=masks,
                                    pointcloud=pointcloud,
                                    scene_name=scene_name)
        

        # 6. Run extractor
        
        features = features_extractor.extract_features(topk=ctx.search3d.mask_features.top_k, 
                                                        multi_level_expansion_ratio = ctx.search3d.mask_features.multi_level_expansion_ratio,
                                                        num_levels=ctx.search3d.mask_features.num_of_levels, 
                                                        num_random_rounds=ctx.search3d.mask_features.num_random_rounds,
                                                        num_selected_points=ctx.search3d.mask_features.num_selected_points,
                                                        save_crops=ctx.output.save_crops,
                                                        out_folder=ctx.output.mask_features_save_dir)

        # 7. Save features
        filename = scene_name+ctx.output.mask_features_suffix
        output_path = os.path.join(ctx.output.mask_features_save_dir, filename)
        torch.save(features, output_path)
        print(f"[INFO]: Masks features for scene {scene_name} saved to {output_path}.")
        #break
    
if __name__ == "__main__":
    main()