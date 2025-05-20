import numpy as np
from PIL import Image
#import open3d as o3d
import imageio
import torch
import math
import os
import trimesh
import pdb
def get_number_of_images(poses_path, id_zfill=0, poses_ext=".txt"):
    i = 0
    while(os.path.isfile(os.path.join(poses_path, str(i).zfill(id_zfill) + poses_ext))): i += 1
    return i

class Camera:
    def __init__(self, 
                 intrinsic_path, 
                 intrinsic_resolution, 
                 poses_path, 
                 depths_path, 
                 extension_depth, 
                 depth_scale,
                 poses_extension,
                 id_zfill=0):
        if intrinsic_path.endswith(".npy"):
            self.intrinsic = np.load(intrinsic_path)[:3, :3]
        else:
            self.intrinsic = np.loadtxt(intrinsic_path)[:3, :3]
        self.intrinsic_original_resolution = intrinsic_resolution
        self.poses_path = poses_path
        self.depths_path = depths_path
        self.extension_depth = extension_depth
        self.depth_scale = depth_scale
        self.poses_extension = poses_extension
        self.id_zfill = id_zfill
    
    def get_adapted_intrinsic(self, desired_resolution):
        '''Get adjusted camera intrinsics.'''
        if self.intrinsic_original_resolution == desired_resolution:
            return self.intrinsic
        
        resize_width = int(math.floor(desired_resolution[1] * float(
                        self.intrinsic_original_resolution[0]) / float(self.intrinsic_original_resolution[1])))
        
        adapted_intrinsic = self.intrinsic.copy()
        adapted_intrinsic[0, 0] *= float(resize_width) / float(self.intrinsic_original_resolution[0])
        adapted_intrinsic[1, 1] *= float(desired_resolution[1]) / float(self.intrinsic_original_resolution[1])
        adapted_intrinsic[0, 2] *= float(desired_resolution[0] - 1) / float(self.intrinsic_original_resolution[0] - 1)
        adapted_intrinsic[1, 2] *= float(desired_resolution[1] - 1) / float(self.intrinsic_original_resolution[1] - 1)
        return adapted_intrinsic
    
    def load_poses(self, indices):
        path = os.path.join(self.poses_path, str(0).zfill(self.id_zfill) + self.poses_extension)
        if self.poses_extension==".npy":
            shape = np.linalg.inv(np.load(path))[:3, :].shape
        else:
            shape = np.linalg.inv(np.loadtxt(path))[:3, :].shape

        poses = np.zeros((len(indices), shape[0], shape[1]))
        for i, idx in enumerate(indices):
            path = os.path.join(self.poses_path, str(idx).zfill(self.id_zfill) + self.poses_extension)
            if self.poses_extension==".npy":
                poses[i] = np.linalg.inv(np.load(path))[:3, :]
            else:
                poses[i] = np.linalg.inv(np.loadtxt(path))[:3, :]
        return poses

class Images:
    def __init__(self, 
                 images_path, 
                 extension, 
                 indices,
                 id_zfill=0):
        self.images_path = images_path
        self.extension = extension
        self.indices = indices
        self.id_zfill = id_zfill
        self.images = self.load_images(indices)
    
    def load_images(self, indices):
        images = []
        for idx in indices:
            img_path = os.path.join(self.images_path, str(idx).zfill(self.id_zfill) + self.extension)
            images.append(Image.open(img_path).convert("RGB"))
        return images
    def get_as_np_list(self):
        images = []
        for i in range(len(self.images)):
            images.append(np.asarray(self.images[i]))
        return images
    
class InstanceMasks3D:
    def __init__(self, masks_path, topk=None):
        if(topk != None):
            self.masks = torch.load(masks_path)[topk].astype(bool)
            self.num_masks = topk
        else:
            self.masks = torch.load(masks_path).astype(bool)
            self.num_masks = self.masks.shape[1]
    
    
class PointCloud:
    def __init__(self, 
                 point_cloud_path):
        pcd = trimesh.load(point_cloud_path, process=False)
        self.points = np.asarray(pcd.vertices)
        self.num_points = self.points.shape[0]
    
    def get_homogeneous_coordinates(self):
        return np.append(self.points, np.ones((self.num_points,1)), axis = -1)
    