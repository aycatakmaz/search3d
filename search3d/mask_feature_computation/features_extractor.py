import os
import PIL
from tqdm import tqdm
import numpy as np
import imageio
import jax
import jax.numpy as jnp
jax.devices()
import ml_collections

import search3d.mask_feature_computation.big_vision_siglip.big_vision.pp.builder as pp_builder
import search3d.mask_feature_computation.big_vision_siglip. big_vision.pp.ops_general
import search3d.mask_feature_computation.big_vision_siglip.big_vision.pp.ops_image
import search3d.mask_feature_computation.big_vision_siglip.big_vision.pp.ops_text
import search3d.mask_feature_computation.big_vision_siglip.big_vision.models.proj.image_text.two_towers as model_mod

import torch
from search3d.data.load import Camera, InstanceMasks3D, PointCloud
from search3d.mask_feature_computation.utils import mask2box_multi_level

class PointProjector:
    def __init__(self, camera: Camera, 
                 point_cloud: PointCloud, 
                 masks: InstanceMasks3D, 
                 vis_threshold, 
                 indices, ctx):
        self.ctx = ctx
        self.vis_threshold = vis_threshold
        self.indices = indices
        self.camera = camera
        self.point_cloud = point_cloud
        self.masks = masks
        self.visible_points_in_view_in_mask, self.visible_points_view, self.projected_points, self.resolution = self.get_visible_points_in_view_in_mask()
        
        
    def get_visible_points_view(self):
        vis_threshold = self.vis_threshold
        indices = self.indices
        depth_scale = self.camera.depth_scale
        poses = self.camera.load_poses(indices)
        X = self.point_cloud.get_homogeneous_coordinates()
        n_points = self.point_cloud.num_points
        depths_path = self.camera.depths_path 

        resolution = self.ctx.data.images.target_shape
        height = resolution[0]
        width = resolution[1]
        intrinsic = self.camera.get_adapted_intrinsic(resolution)

        projected_points = np.zeros((len(indices), n_points, 2), dtype = int)
        visible_points_view = np.zeros((len(indices), n_points), dtype = bool)
        print(f"[INFO] Computing the visible points in each view.")
        
        for i, idx in tqdm(enumerate(indices)): # for each view
            # *******************************************************************************************************************
            # STEP 1: get the projected points
            # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
            projected_points_not_norm = (intrinsic @ poses[i] @ X.T).T
            # Get the mask of the points which have a non-null third coordinate to avoid division by zero
            mask = (projected_points_not_norm[:, 2] != 0) # don't do the division for point with the third coord equal to zero
            # Get non homogeneous coordinates of valid points (2D in the image)
            projected_points[i][mask] = np.column_stack([[projected_points_not_norm[:, 0][mask]/projected_points_not_norm[:, 2][mask], 
                    projected_points_not_norm[:, 1][mask]/projected_points_not_norm[:, 2][mask]]]).T
            
            # *******************************************************************************************************************
            # STEP 2: occlusion computation
            # Load the depth from the sensor
            depth_path = os.path.join(depths_path, str(idx).zfill(int(self.ctx.data.id_zfill)) + '.png')

            sensor_depth = imageio.v2.imread(depth_path) / depth_scale                   
            sensor_depth_max = sensor_depth.max()
            sensor_depth = np.array(PIL.Image.fromarray(sensor_depth).resize((width, height)))
            sensor_depth[sensor_depth<0] = 0
            sensor_depth[sensor_depth>sensor_depth_max] = sensor_depth_max

            inside_mask = (projected_points[i,:,0] >= 0) * (projected_points[i,:,1] >= 0) \
                                * (projected_points[i,:,0] < width) \
                                * (projected_points[i,:,1] < height)
            pi = projected_points[i].T
            # Depth of the points of the pointcloud, projected in the i-th view, computed using the projection matrices
            point_depth = projected_points_not_norm[:,2]
            # Compute the visibility mask, true for all the points which are visible from the i-th view
            visibility_mask = (np.abs(sensor_depth[pi[1][inside_mask], pi[0][inside_mask]]
                                        - point_depth[inside_mask]) <= \
                                        vis_threshold).astype(bool)
            inside_mask[inside_mask == True] = visibility_mask
            visible_points_view[i] = inside_mask
        return visible_points_view, projected_points, resolution
    
    def get_bbox(self, mask, view):
        if(self.visible_points_in_view_in_mask[view][mask].sum()!=0):
            true_values = np.where(self.visible_points_in_view_in_mask[view, mask])
            valid = True
            t, b, l, r = true_values[0].min(), true_values[0].max()+1, true_values[1].min(), true_values[1].max()+1 
        else:
            valid = False
            t, b, l, r = (0,0,0,0)
        return valid, (t, b, l, r)
    
    def get_visible_points_in_view_in_mask(self):
        masks = self.masks
        num_view = len(self.indices)
        visible_points_view, projected_points, resolution = self.get_visible_points_view()
        visible_points_in_view_in_mask = np.zeros((num_view, masks.num_masks, resolution[0], resolution[1]), dtype=bool)
        print(f"[INFO] Computing the visible points in each view in each mask.")
        for i in tqdm(range(num_view)):
            for j in range(masks.num_masks):
                visible_masks_points = (masks.masks[:,j] * visible_points_view[i]) > 0
                proj_points = projected_points[i][visible_masks_points]
                if(len(proj_points) != 0):
                    visible_points_in_view_in_mask[i][j][proj_points[:,1], proj_points[:,0]] = True
        self.visible_points_in_view_in_mask = visible_points_in_view_in_mask
        self.visible_points_view = visible_points_view
        self.projected_points = projected_points
        self.resolution = resolution
        return visible_points_in_view_in_mask, visible_points_view, projected_points, resolution
    
    def get_top_k_indices_per_mask(self, k):
        num_points_in_view_in_mask = self.visible_points_in_view_in_mask.sum(axis=2).sum(axis=2)
        topk_indices_per_mask = np.argsort(-num_points_in_view_in_mask, axis=0)[:k,:].T
        return topk_indices_per_mask
    
class FeaturesExtractor:
    def __init__(self, 
                 ctx,
                 device):
        
        self.ctx = ctx
        self.device = device

        self.predictor_sam = None
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        VARIANT = self.ctx.external.siglip_variant
        RES = self.ctx.external.siglip_res
        TXTVARIANT = self.ctx.external.siglip_txtvariant
        self.EMBDIM = EMBDIM = self.ctx.external.siglip_embdim
        SEQLEN = self.ctx.external.siglip_seqlen
        VOCAB = self.ctx.external.siglip_vocab
        TOKENIZER = self.ctx.external.siglip_tokenizer
        CKPT_PATH = self.ctx.external.siglip_checkpoint_path

        #"""
        print(VARIANT, RES, TXTVARIANT, EMBDIM, SEQLEN, VOCAB, TOKENIZER, CKPT_PATH)
        model_cfg = ml_collections.ConfigDict()
        model_cfg.image_model = 'vit'  # TODO(lbeyer): remove later, default
        model_cfg.text_model = 'proj.image_text.text_transformer'  # TODO(lbeyer): remove later, default
        model_cfg.image = dict(variant=VARIANT, pool_type='map')
        model_cfg.text = dict(variant=TXTVARIANT, vocab_size=VOCAB)
        model_cfg.out_dim = (None, EMBDIM)  # (image_out_dim, text_out_dim)
        model_cfg.bias_init = -10.0
        model_cfg.temperature_init = 10.0

        print("[INFO] Initializing SigLIP... (might take up to 1 minute)")
        self.siglip_model = model_mod.Model(**model_cfg)
        init_params = None
        self.siglip_params = model_mod.load(init_params, CKPT_PATH, model_cfg)

        self.siglip_preprocess = pp_builder.get_preprocess_fn(f'resize({RES})|value_range(-1, 1)', log_data=False)
        self.siglip_preprocess_text = pp_builder.get_preprocess_fn(f'tokenize(max_len={SEQLEN}, model="{TOKENIZER}", eos="sticky", pad_value=1, inkey="text", load_from_file_path="{self.ctx.external.siglip_tokenizer_load_from_file_path}")')
        
        print("[INFO] Initialized SigLIP!")
        #"""

    def set_scene(self, camera, images, masks, pointcloud, scene_name="scene"):
        self.camera = camera
        self.images = images
        self.scene_name = scene_name
        print("[INFO] Initializing PointProjector...")
        self.point_projector = PointProjector(camera, pointcloud, masks, self.ctx.search3d.mask_features.vis_threshold, images.indices, ctx=self.ctx)
        print("[INFO] Initialized PointProjector!")

    def extract_features(self, topk, multi_level_expansion_ratio, num_levels, num_random_rounds, num_selected_points, save_crops, out_folder):
        if(save_crops):
            out_folder = os.path.join(out_folder, "crops", self.scene_name)
            os.makedirs(out_folder, exist_ok=True)
                            
        topk_indices_per_mask = self.point_projector.get_top_k_indices_per_mask(topk)
        
        num_masks = self.point_projector.masks.num_masks
        mask_clip = np.zeros((num_masks, self.EMBDIM)) #initialize mask clip
        
        np_images = self.images.get_as_np_list()
        for mask in tqdm(range(num_masks)): # for each mask 
            images_crops = []
            for view_count, view in enumerate(topk_indices_per_mask[mask]): # for each view
                
                # Get original mask points coordinates in 2d images
                point_coords = np.transpose(np.where(self.point_projector.visible_points_in_view_in_mask[view][mask] == True))
                if (point_coords.shape[0] > 0):
                    
                    
                    assert self.predictor_sam is None
                    best_mask = np.zeros_like(np_images[view])[:,:,0]
                    best_mask[point_coords[:,0],point_coords[:,1]]=1

                    # save mask in the full image
                    if(save_crops):
                        imageio.imwrite(os.path.join(out_folder, f"mask{mask}_{view}.png"), best_mask*255)

                    # MULTI LEVEL CROPS
                    for level in range(num_levels):
                        # get the bbox and corresponding crops
                        x1, y1, x2, y2 = mask2box_multi_level(torch.from_numpy(best_mask), level, multi_level_expansion_ratio)
                        cropped_img = self.images.images[view].crop((x1, y1, x2, y2))
                        
                        if(save_crops):
                            cropped_img.save(os.path.join(out_folder, f"crop{mask}_{view}_{level}.png"))

                        images_crops.append(cropped_img)

            if(len(images_crops) > 0):
                
                images_processed = np.array([self.siglip_preprocess({'image': np.array(img_crop)})['image'] for img_crop in images_crops])
                zimg, _, siglip_out = self.siglip_model.apply({'params': self.siglip_params}, images_processed, None)

                zimg_normalized = zimg / jnp.linalg.norm(zimg, axis=-1, keepdims=True) # not needed actually, they are already l2-normalized
                mask_clip[mask] = np.array(zimg_normalized).mean(axis=0)
                    
        return mask_clip
        
    