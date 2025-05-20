import numpy as np
import torch
import pdb
import cv2
from PIL import Image

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
    x1, y1, x2 , y2  = mask2box(mask)
    if level == 0:
        return x1, y1, x2 , y2
    shape = mask.shape
    x_exp = int(abs(x2- x1)*expansion_ratio) * level
    y_exp = int(abs(y2-y1)*expansion_ratio) * level
    return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)

@torch.no_grad()
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

def draw_ellipse(image, points, scale=5, color=(0, 0, 255), thickness=3):
    # image is a PIL image, convert it to cv2 image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    ellipse_cent_x, ellipse_cent_y = np.mean(points, axis=0)[::-1]
    ellipse_width, ellipse_height = scale*np.std(points[:, 1]), scale*np.std(points[:, 0])
    ellipse_angle = 0 #fitted_ellipse[2]
    new_ellipse = ((ellipse_cent_x, ellipse_cent_y), (ellipse_width, ellipse_height), ellipse_angle)
    res = cv2.ellipse(img.copy(), new_ellipse, color, thickness)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(res)

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()
    