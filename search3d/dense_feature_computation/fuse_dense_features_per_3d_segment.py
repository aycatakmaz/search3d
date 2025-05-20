import jax
import jax.numpy as jnp
jax.devices()
import ml_collections

import search3d.mask_feature_computation.big_vision_siglip.big_vision.pp.builder as pp_builder
import search3d.mask_feature_computation.big_vision_siglip. big_vision.pp.ops_general
import search3d.mask_feature_computation.big_vision_siglip.big_vision.pp.ops_image
import search3d.mask_feature_computation.big_vision_siglip.big_vision.pp.ops_text
import search3d.mask_feature_computation.big_vision_siglip.big_vision.models.proj.image_text.two_towers as model_mod

import numpy as np
import torch
from PIL import Image
import os
import imageio
from glob import glob
from tqdm import tqdm
from fusion_utils import PointCloudToImageMapper, adjust_intrinsic
import trimesh
import hydra
from omegaconf import DictConfig

class DenseFeatureFusion3D():
    def __init__(self, ctx, dense_feature_extractor):
        self.ctx = ctx
        self.img_dim = ctx.data.images.img_dim # (width, height) 
        self.target_shape = (ctx.data.images.img_dim[1], ctx.data.images.img_dim[0]) # (height, width)
        self.depth_scale = float(ctx.data.depths.depth_scale)
        self.visibility_threshold = ctx.search3d.dense_features.vis_threshold
        self.cut_num_pixel_boundary = ctx.search3d.dense_features.cut_num_pixel_boundary
        self.frequency = ctx.search3d.dense_features.frequency

        self.scans_path = ctx.data.scans_path

        self.out_dir = ctx.output.segment_features_save_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.segments_dir = ctx.data.segments.segments_dir

        self.dense_feature_extractor = dense_feature_extractor
        self.feat_dim = self.dense_feature_extractor.feat_dim
    
    def set_scene(self, scene_name):
        self.scene_name = scene_name

        if os.path.exists(os.path.join(self.out_dir, scene_name+self.ctx.output.segment_features_suffix)):
            print("[WARNING]", os.path.join(self.out_dir, scene_name+self.ctx.output.segment_features_suffix), 'already done!')
            #pdb.set_trace()
        
        self.scene_dir = os.path.join(self.scans_path, scene_name)

        self.ply_path = os.path.join(self.scene_dir, scene_name + self.ctx.data.point_cloud_suffix)

        self.images_dir = os.path.join(self.scene_dir, self.ctx.data.images.images_path)
        img_paths_full = sorted(glob(os.path.join(self.images_dir, '*')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.img_paths = img_paths_full[::self.frequency]

        self.poses_dir = os.path.join(self.scene_dir, self.ctx.data.camera.poses_path)
        pose_paths_full = sorted(glob(os.path.join(self.poses_dir, '*')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.pose_paths = pose_paths_full[::self.frequency]

        self.depths_dir = os.path.join(self.scene_dir, self.ctx.data.depths.depths_path)
        depth_paths_full = sorted(glob(os.path.join(self.depths_dir, '*')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = depth_paths_full[::self.frequency]

        intrinsics_path = os.path.join(self.scene_dir, self.ctx.data.camera.intrinsic_path)
        orig_intrinsic = np.loadtxt(intrinsics_path)[:3, :3]
        self.intrinsic = adjust_intrinsic(orig_intrinsic.copy(), intrinsic_image_dim=self.ctx.data.camera.intrinsic_resolution, image_dim=self.target_shape)

        # calculate image pixel-3D points correspondances
        self.point2img_mapper = PointCloudToImageMapper(
            image_dim=self.img_dim, intrinsics=self.intrinsic,
            visibility_threshold=self.visibility_threshold,
            cut_bound=self.cut_num_pixel_boundary)
        
        self.ply_path = os.path.join(self.scene_dir, scene_name + self.ctx.data.point_cloud_suffix)
        self.pcd = trimesh.load(self.ply_path, process=False)
        self.coords = np.asarray(self.pcd.vertices)
        
        self.segments_tree_dict_pt_path = os.path.join(self.segments_dir, scene_name + self.ctx.data.segments.segments_tree_dict_pt_suffix)
        self.segments_tree_dict = torch.load(self.segments_tree_dict_pt_path)
        if 'num_points' in self.segments_tree_dict:
            del self.segments_tree_dict['num_points']

    def process_one_scene(self, scene_name):
        # set scene paths and parameters
        self.set_scene(scene_name)
        n_points = len(self.coords)

        if self.ctx.gpu.optimize_gpu_usage:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')

        segments_unrolled = torch.from_numpy(np.concatenate([segments for segments in self.segments_tree_dict.values()], axis=1)).to(device) #(237360, 1411)
        segment_features_unrolled = torch.zeros((segments_unrolled.shape[1], self.feat_dim), device=device) #torch.Size([1411, 1024]) ## num_segments: 1885 (0.03, 20) and 1411 (0.05, 100)

        if self.ctx.search3d.dense_features.vlm_model_type == "siglip":
            masks_2d_all_frames = []
            self.dense_feature_extractor.move_mask_generator_to_device(torch.device("cuda"))
            for img_id, img_path in enumerate(tqdm(self.img_paths)):
                masks_2d_for_img = self.dense_feature_extractor.compute_masks_2d_for_img(img_path, min_num_pix=75, target_shape=self.target_shape) #(46, 480, 640)
                if masks_2d_for_img is None or len(torch.nonzero(masks_2d_for_img.sum(dim=0)))<0:
                    masks_2d_for_img = torch.ones((1, *self.target_shape), dtype=torch.bool)
                    print("No masks found for frame_id {} - setting an all-ones mask.".format(str(img_id)))
                masks_2d_all_frames.append(masks_2d_for_img)
            self.dense_feature_extractor.move_mask_generator_to_device(torch.device("cpu"))

        ################ Feature Fusion ###################
        for img_id, img_path in enumerate(tqdm(self.img_paths)):
            # load pose
            pose = np.loadtxt(self.pose_paths[img_id])

            # load depth and convert to meters
            depth = imageio.v2.imread(self.depth_paths[img_id]) / self.depth_scale     
            if depth.shape != tuple(self.ctx.data.images.target_shape): # only if we need to resize the depth images, we resize them to match the target shape    
                depth_max = depth.max()
                depth = np.array(Image.fromarray(depth).resize((self.ctx.data.images.target_shape[1], self.ctx.data.images.target_shape[0])))
                depth[depth<0] = 0
                depth[depth>depth_max] = depth_max

            # calculate the 3d-2d mapping based on the depth
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = self.point2img_mapper.compute_mapping(pose, self.coords, depth)
            if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
                continue

            mapping = torch.from_numpy(mapping).to(device) #torch.Size([237360, 4])
            
            mask = mapping[:, 3]

            if self.ctx.search3d.dense_features.vlm_model_type == "siglip":
                feat_2d = self.dense_feature_extractor.compute_dense_features(img_path, masks_2d=masks_2d_all_frames[img_id], target_shape=self.target_shape).permute(2, 0, 1).to(device) ###.to(device)
            else:
                feat_2d = self.dense_feature_extractor.compute_dense_features(img_path, target_shape=self.target_shape).permute(2, 0, 1).to(device) ###.to(device)

            feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0) #torch.Size([237360, 1024])
            
            non_zero_mask_point_indices = torch.where(mask!=0)[0] #tensor([  3135,   3136,   3137,  ..., 198892, 200191, 200192], device='cuda:0')
            for mask_point_idx in non_zero_mask_point_indices:
                segment_features_unrolled[segments_unrolled[mask_point_idx], :] += feat_2d_3d[mask_point_idx, :] #no normalization though?

        segment_features_unrolled_norms = torch.norm(segment_features_unrolled, dim=1)
        segment_features_unrolled[segment_features_unrolled_norms>1e-5, :] /= segment_features_unrolled_norms[segment_features_unrolled_norms>1e-5].unsqueeze(dim=1) # normalize per-segment features

        # roll back the features into a dictionary: keys: inst_mask_id, each element is a tensor of shape (n_segments, feat_dim) for each instance including all the segments for a given instance
        segment_features_dict = {}
        segment_start_id = 0
        for inst_mask_id, inst_segments in self.segments_tree_dict.items():
            segment_features_dict[inst_mask_id] = segment_features_unrolled[segment_start_id:segment_start_id+inst_segments.shape[1], :].half().cpu()
            segment_start_id += inst_segments.shape[1]
        torch.save({"segment_features": segment_features_dict},  os.path.join(self.out_dir, scene_name + self.ctx.output.segment_features_suffix))
        print("[INFO] Saved the fused features for the scene segments: ", scene_name, "to", os.path.join(self.out_dir, scene_name + self.ctx.output.segment_features_suffix))
            
        
@hydra.main(config_path="../configs", config_name="search3d_scannet200_eval", version_base=None) 
def main(ctx: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    
    # uncomment this to test locally from this folder
    # dense_feature_extractor = DenseFeatureExtractor(ctx=ctx, config_root="semantic_sam/configs")
    
    if ctx.search3d.dense_features.vlm_model_type == "siglip":
        print("[INFO] Initializing the DenseFeatureExtractorSigLIP with SigLIP...")
        from search3d.dense_feature_computation.compute_dense_image_features_siglip import DenseFeatureExtractorSigLIP
        dense_feature_extractor = DenseFeatureExtractorSigLIP(ctx=ctx, semantic_sam_config_root="dense_feature_computation/semantic_sam/configs")
        print("[INFO] Initialized the DenseFeatureExtractorSigLIP with SigLIP.")
    else:
        raise NotImplementedError("The dense feature extractor model type is not implemented yet, available options: [open_clip, siglip]")
    dense_feature_fusion3d = DenseFeatureFusion3D(ctx=ctx, dense_feature_extractor=dense_feature_extractor)
    
    if len(ctx.data.segments.segments_path)>0:
        segment_path_files = [ctx.data.segments.segments_path]
        scene_names = [os.path.basename(segment_path_file)[:-len(ctx.data.segments.segments_tree_dict_suffix)] for segment_path_file in segment_path_files]
    else:
        segment_path_files = sorted(glob(os.path.join(ctx.data.segments.segments_dir, "*"+ctx.data.segments.segments_tree_dict_suffix)))
        scene_names = [os.path.basename(segment_path_file)[:-len(ctx.data.segments.segments_tree_dict_suffix)] for segment_path_file in segment_path_files]
    
    for scene_name in scene_names:
        dense_feature_fusion3d.process_one_scene(scene_name)

if __name__=="__main__":
    main()