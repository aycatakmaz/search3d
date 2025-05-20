import numpy as np
import torch
from search3d.dense_feature_computation.semantic_sam.semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator
import open_clip
import random

def initialize_semantic_sam_mask_generator(device, semantic_sam_model_type, semantic_sam_checkpoint_path, config_root="semantic_sam/configs"):
    granularity_level = [4,5,6]
    mask_generator = SemanticSamAutomaticMaskGenerator(build_semantic_sam(model_type=semantic_sam_model_type, 
                                                                      ckpt=semantic_sam_checkpoint_path,
                                                                      config_root=config_root), 
                                                                      level=granularity_level) # model_type: 'L' / 'T', depends on your checkpint
    return mask_generator

def get_open_clip_model(open_clip_model_name, open_clip_pretrained_dataset_name, device):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(open_clip_model_name, open_clip_pretrained_dataset_name)
    clip_model.to(device)
    clip_model.eval()
    return clip_model, preprocess


def get_semantic_sam_mask(mask_generator, image_path, min_num_pix=75, target_shape=None):
    original_image, input_image = prepare_image(image_pth=image_path)
    # get semantic sam masks for the granularity level specified while building the mask generator
    masks_dict = mask_generator.generate(input_image)

    masks = np.asarray([mask['segmentation'] for mask in masks_dict])
    # we get mask areas to sort the masks based on their area (descending)
    mask_areas = np.asarray([mask['area'] for mask in masks_dict])
    sorted_indices = np.argsort(mask_areas)[::-1]
    sorted_masks = masks[sorted_indices]

    new_mask_image_merged = np.zeros_like(input_image[0,:,:].cpu())

    for i, mask in enumerate(sorted_masks):
        new_mask_image_merged[mask] = i+1

    new_masks = []
    unique_instances = np.unique(new_mask_image_merged)
    for unique_instance in unique_instances:
        if unique_instance == 0:
            continue
        curr_mask = new_mask_image_merged == unique_instance
        if curr_mask.sum() < min_num_pix:  # filter out small masks
            continue
        new_masks.append(curr_mask)

    new_masks = torch.from_numpy(np.asarray(new_masks))
    if new_masks.ndim == 2:
        new_masks = new_masks.unsqueeze(dim=0)

    if target_shape is not None:
        new_masks = torch.nn.functional.interpolate(new_masks.unsqueeze(dim=0).float(), [target_shape[0], target_shape[1]], mode="nearest").squeeze(dim=0).numpy()>0.5

    return new_masks

def mask2box(mask: torch.Tensor):
    row = torch.nonzero(mask.sum(axis=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min().item()
    x2 = row.max().item()
    col = np.nonzero(mask.sum(axis=1))[:, 0]
    y1 = col.min().item()
    y2 = col.max().item()
    return x1, y1, x2 + 1, y2 + 1

def mask2box_multi_level(mask: torch.Tensor, level, expansion_ratio):
    x1, y1, x2, y2  = mask2box(mask)
    if level == 0:
        return x1, y1, x2, y2
    shape = mask.shape
    x_exp = int(abs(x2- x1)*expansion_ratio) * level
    y_exp = int(abs(y2-y1)*expansion_ratio) * level
    return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)

def run_sam(image_size, num_random_rounds, num_selected_points, point_coords, predictor_sam):
    best_score = 0
    best_mask = np.zeros_like(image_size, dtype=bool)
    
    point_coords_new = np.zeros_like(point_coords)
    point_coords_new[:,0] = point_coords[:,1]
    point_coords_new[:,1] = point_coords[:,0]
    
    # Get only a random subsample of them for num_random_rounds times and choose the mask with highest confidence score
    for i in range(num_random_rounds):
        np.random.shuffle(point_coords_new)
        masks, scores, logits = predictor_sam.predict(
            point_coords=point_coords_new[:num_selected_points],
            point_labels=np.ones(point_coords_new[:num_selected_points].shape[0]),
            multimask_output=False,
        )  
        if scores[0] > best_score:
            best_score = scores[0]
            best_mask = masks[0]
            
    return best_mask

def set_seeds(seed=44):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)