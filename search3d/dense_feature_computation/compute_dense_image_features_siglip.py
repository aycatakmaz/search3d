import jax
import jax.numpy as jnp
jax.devices()
import ml_collections

import search3d.mask_feature_computation.big_vision_siglip.big_vision.pp.builder as pp_builder
import search3d.mask_feature_computation.big_vision_siglip.big_vision.models.proj.image_text.two_towers as model_mod

import os
import pdb
import numpy as np
import torch

from search3d.dense_feature_computation.utils import initialize_semantic_sam_mask_generator, get_semantic_sam_mask, mask2box, mask2box_multi_level
from search3d.dense_feature_computation.utils import set_seeds

import matplotlib.pyplot as plt
from PIL import Image
import cv2
import hydra
from omegaconf import DictConfig


class DenseFeatureExtractorSigLIP():
    def __init__(self, ctx, semantic_sam_config_root="semantic_sam/configs"):
        self.ctx = ctx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seeds(self.ctx.search3d.dense_features.seed)
        self.vlm_model_type = self.ctx.search3d.dense_features.vlm_model_type
        self.initialize_models(semantic_sam_config_root=semantic_sam_config_root) # sets feature dimension and preprocessing functions

    def initialize_models(self, semantic_sam_config_root="semantic_sam/configs"):
        if self.vlm_model_type == "siglip":
            print("[INFO] Initializing the SIGLIP model...")
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            VARIANT = self.ctx.external.siglip_variant
            RES = self.ctx.external.siglip_res
            TXTVARIANT = self.ctx.external.siglip_txtvariant
            self.feat_dim = EMBDIM = self.ctx.external.siglip_embdim
            SEQLEN = self.ctx.external.siglip_seqlen
            VOCAB = self.ctx.external.siglip_vocab
            TOKENIZER = self.ctx.external.siglip_tokenizer
            CKPT_PATH = self.ctx.external.siglip_checkpoint_path

            print(VARIANT, RES, TXTVARIANT, EMBDIM, SEQLEN, VOCAB, TOKENIZER, CKPT_PATH)
            model_cfg = ml_collections.ConfigDict()
            model_cfg.image_model = 'vit' 
            model_cfg.text_model = 'proj.image_text.text_transformer' 
            model_cfg.image = dict(variant=VARIANT, pool_type='map')
            model_cfg.text = dict(variant=TXTVARIANT, vocab_size=VOCAB)
            model_cfg.out_dim = (None, EMBDIM)  # (image_out_dim, text_out_dim)
            model_cfg.bias_init = -10.0
            model_cfg.temperature_init = 10.0

            print("[INFO] Initializing SigLIP... (might take up to 1 minute)")
            self.siglip_model = model_mod.Model(**model_cfg)
            self.init_params = None
            self.siglip_params = model_mod.load(self.init_params, CKPT_PATH, model_cfg)

            self.siglip_preprocess = pp_builder.get_preprocess_fn(f'resize({RES})|value_range(-1, 1)')
            self.siglip_preprocess_text = pp_builder.get_preprocess_fn(f'tokenize(max_len={SEQLEN}, model="{TOKENIZER}", eos="sticky", pad_value=1, inkey="text", load_from_file_path="{self.ctx.external.siglip_tokenizer_load_from_file_path}")')

            print("[INFO] CLIP model is initialized.")

        else:
            raise NotImplementedError

        # get SAM masks
        print("[INFO] Initializing the SemanticSAM mask generator...")
        self.mask_generator = initialize_semantic_sam_mask_generator(self.device, self.ctx.search3d.dense_features.mask_generator_2d.model_type, self.ctx.search3d.dense_features.mask_generator_2d.model_ckpt_path, config_root=semantic_sam_config_root)
        print("[INFO] SemanticSAM mask generator is initialized.")

    def siglip_features_for_image_list(self, img_rois):
        img_rois = np.asarray([self.siglip_preprocess({'image': img_roi})['image'] for img_roi in img_rois])
        zimg_rois, _, _ = self.siglip_model.apply({'params': self.siglip_params}, img_rois, None)
        zimg_normalized = zimg_rois / jnp.linalg.norm(zimg_rois, axis=-1, keepdims=True)
        return zimg_normalized


    def aggregate_dense_features(self, masks, img, target_shape=(480, 640)):
        with torch.no_grad():
            LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = img.shape[0], img.shape[1]
            if (target_shape[0] != LOAD_IMG_HEIGHT) or (target_shape[1] != LOAD_IMG_WIDTH):
                raise NotImplementedError("Our code currently assumes that the target shape is the same as the input image shape.")

            #outfeat =  np.zeros((LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, self.feat_dim)) #device?            
            img_rois = []

            for mask_idx, mask in enumerate(masks):
                if mask.sum() == 0:
                    img_rois.append(img)
                else:
                    expanded_box = mask2box_multi_level(mask, level=1, expansion_ratio=self.ctx.search3d.dense_features.single_level_expansion_ratio)
                    x1, y1, x2, y2 = expanded_box

                    img_roi = img[y1:y2, x1:x2, :]
                    img_rois.append(img_roi)

            assert len(img_rois) == len(masks)

            siglip_feats = np.zeros((len(img_rois), self.feat_dim))

            siglip_batch_size = self.ctx.external.siglip_max_batch_size
            for crop_idx in range(0, len(img_rois), siglip_batch_size):
                siglip_feats[crop_idx:crop_idx+siglip_batch_size] = self.siglip_features_for_image_list(img_rois[crop_idx:crop_idx+siglip_batch_size])
            
            del img_rois

            siglip_feats = siglip_feats.astype(np.float32)
            outfeat = np.zeros((LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, self.feat_dim), dtype=np.float32)
            for mask_idx, mask in enumerate(masks):
                nonzero_inds = np.argwhere(np.asarray(mask))
                outfeat[nonzero_inds[:, 0], nonzero_inds[:, 1]] += siglip_feats[mask_idx, :] # these are already normalized
                outfeat[nonzero_inds[:, 0], nonzero_inds[:, 1]] = outfeat[nonzero_inds[:, 0], nonzero_inds[:, 1]] / np.linalg.norm(outfeat[nonzero_inds[:, 0], nonzero_inds[:, 1]], axis=-1, keepdims=True)

            outfeat = torch.from_numpy(np.array(outfeat))
            
        return outfeat # returns #torch.Size([480, 640, 1024])


    def compute_masks_2d_for_img(self, img_path, min_num_pix=75, target_shape=(480, 640)):
        masks = get_semantic_sam_mask(self.mask_generator, img_path, min_num_pix=min_num_pix, target_shape=target_shape)
        masks = torch.from_numpy(masks)
        return masks
    
    def move_mask_generator_to_device(self, device):
        self.mask_generator.predictor.model.to(device)
        torch.cuda.empty_cache()

    def compute_dense_features(self, img_path, masks_2d=None, target_shape=(480, 640), min_num_pix=75):
        # load img
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        if masks_2d is None:
            masks = self.compute_masks_2d_for_img(img_path, min_num_pix=min_num_pix, target_shape=target_shape) #(46, 480, 640)
        else:
            masks = masks_2d

        dense_features = self.aggregate_dense_features(masks, img, target_shape=target_shape)

        return dense_features
    

@hydra.main(config_path="../configs", config_name="search3d_scannet200_eval", version_base=None) 
def main(ctx: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    print("[INFO] Initializing the DenseFeatureExtractorSigLIP...")
    dense_feature_extractor = DenseFeatureExtractorSigLIP(ctx=ctx)
    print("[INFO] Initialized the DenseFeatureExtractorSigLIP.")

    img_path = "/home/ayca/concept-fusion-new/concept-fusion/examples/scannet_chairs.jpeg" #"chair_in_a_kitchen.jpg"

    import time
    start = time.time()
    dense_features = dense_feature_extractor.compute_dense_features(img_path) #torch.Size([480, 640, 1024])
    end = time.time()
    print("[INFO] Time taken for dense feature computation: ", end-start)
    torch.cuda.empty_cache()

    print("[INFO] Saving the dense features...")
    pdb.set_trace()

if __name__=="__main__":
    main()
