import os
import pdb
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from segmentator import segment_mesh, segment_point, compute_vn
from glob import glob
import trimesh
import open3d as o3d
from copy import deepcopy
from tqdm import tqdm
import json
import open3d as o3d

class SegmentComputation():
    def __init__(self, ctx):
        self.ctx = ctx
        self.kThresh = ctx.search3d.segment_computation.kThresh
        self.segMinVerts = ctx.search3d.segment_computation.segMinVerts

        self.scans_path = ctx.data.scans_path
        self.masks_dir = ctx.data.masks.masks_dir
        self.masks_suffix = ctx.data.masks.masks_suffix

        self.segments_dir = ctx.data.segments.segments_dir
        if not os.path.exists(self.segments_dir):
            os.makedirs(self.segments_dir)
        print(f"[INFO] Segments will be saved in {self.segments_dir}")

        self.segments_tree_dict_suffix = ctx.data.segments.segments_tree_dict_suffix
        self.segments_tree_dict_pt_suffix = ctx.data.segments.segments_tree_dict_pt_suffix

    def compute_segments_all(self):
        print("[INFO] Extracting segments for all instances...")
        masks_paths = sorted(glob(os.path.join(self.masks_dir, "*"+self.masks_suffix)))
        #pdb.set_trace()
        if len(masks_paths) == 0:
            print("[---ERROR---] No masks found in the given directory {}!".format(self.masks_dir))
            return
        
        for masks_path in masks_paths:
            if masks_path.endswith(".npy"):
                masks = np.load(masks_path).T > 0.5
            else:
                masks = torch.load(masks_path).T > 0.5
            
            # load mesh and separate it for each mask
            # this we can do with open3d
            # then we run segmentator by computing the trimesh mesh
            # then we save the dictionary with the segments, where the key is the mask id
            # for the feature computation, instead of aggregating the features over all segments, e.g. (1411, 1024), we store these in a dict structure with instance IDs as the keys
            scene_name = os.path.basename(masks_path)[:-len(self.masks_suffix)]
            scene_dir = os.path.join(self.scans_path, scene_name)

            ply_path = os.path.join(scene_dir, self.ctx.data.point_cloud_suffix.format(scene_name))

            mesh = o3d.io.read_triangle_mesh(ply_path)
            n_points = len(mesh.vertices)
            
            mask_dictionary = {}
            mask_dictionary["kThresh"] = self.kThresh
            mask_dictionary["segMinVerts"] = self.segMinVerts
            mask_dictionary["n_points"] = n_points
            mask_dictionary["masks_segments"] = {} 
            for mask_idx, inst_mask in tqdm(enumerate(masks)):
                inst_mask = inst_mask>0.5
                mask_nonzero = np.argwhere(inst_mask>0.5)[:, 0]

                inst_mesh = mesh.select_by_index(mask_nonzero, cleanup=False)

                inst_vertices = torch.from_numpy(np.asarray(inst_mesh.vertices).astype(np.float32)) #torch.Size([756, 3])
                inst_faces = torch.from_numpy(np.asarray(inst_mesh.triangles).astype(np.int64)) #torch.Size([1357, 3])
                inst_segment_ind = segment_mesh(inst_vertices, inst_faces, kThresh=self.kThresh, segMinVerts=self.segMinVerts) #torch.Size([853])

                ind_unique, ind_counts = np.unique(inst_segment_ind, return_counts=True)
                discard_ind = set(ind_unique[ind_counts<self.ctx.search3d.segment_computation.filter_out_small_segments_num_points])
                if len(discard_ind) > 0:
                    discard_mask = [int(el) in set(discard_ind) for el in inst_segment_ind]
                    inst_segment_ind[discard_mask] = -1
                    coords_with_valid_segment = inst_vertices[inst_segment_ind!=-1]
                    if len(coords_with_valid_segment) == 0:
                        inst_segment_ind = torch.zeros((len(inst_segment_ind))).long()
                        continue
                    coords_with_invalid_segment = inst_vertices[inst_segment_ind==-1]
                    # go over coords_with_zero_segment and assign them to the closest segment
                    replacement_segments = torch.zeros((len(coords_with_invalid_segment))).long()
                    for coord_idx, coord in enumerate(coords_with_invalid_segment):
                        dists = torch.norm(coords_with_valid_segment - coord, dim=1)
                        closest_segment = inst_segment_ind[inst_segment_ind!=-1][torch.argmin(dists)]
                        replacement_segments[coord_idx] = closest_segment
                    inst_segment_ind[inst_segment_ind==-1] = replacement_segments

                segment_id_mapper = {int(segment_id):enum_id for enum_id, segment_id in enumerate(sorted(np.unique(inst_segment_ind)))}
                inst_segments = np.array([segment_id_mapper[int(segment_id)] for segment_id in inst_segment_ind])

                mask_dictionary["masks_segments"][mask_idx] = {}
                mask_dictionary["masks_segments"][mask_idx]["inst_mask"] = mask_nonzero.tolist()
                mask_dictionary["masks_segments"][mask_idx]["segments"] = inst_segments.astype(int).tolist()

                try:
                    assert len(mask_nonzero) == len(inst_segments)
                except:
                    print(f"[ERROR] Length of mask_nonzero and inst_segments do not match for mask_idx: {mask_idx}")
                    pdb.set_trace()

            part_dict_export_path = os.path.join(self.segments_dir, scene_name + self.segments_tree_dict_suffix)
            with open(part_dict_export_path, 'w') as f:
                json.dump(mask_dictionary, f)

            print(f"[INFO] Segment tree saved as a dictionary to: {part_dict_export_path}")
            
            if "scannetpp" in self.ctx.data.scans_path:
                return
            
            segments_tree_dict = {}
            inst_ids = list(int(el) for el in mask_dictionary['masks_segments'].keys())
            for mask_id in tqdm(inst_ids):
                mask = np.asarray(mask_dictionary['masks_segments'][mask_id]["inst_mask"])
                segments = np.asarray(mask_dictionary['masks_segments'][mask_id]["segments"])
                unique_segments = np.unique(segments)
                segments_tree_dict[mask_id] = []
                for unique_segment in unique_segments:
                    unique_segment_indices_in_mask = mask[segments == unique_segment]
                    segment_mask_over_full_pcd = np.zeros((n_points)).astype(bool)
                    segment_mask_over_full_pcd[unique_segment_indices_in_mask] = True
                    segments_tree_dict[mask_id].append(segment_mask_over_full_pcd)

            for mask_id, segments in segments_tree_dict.items():
                segments_tree_dict[mask_id] = np.asarray(segments).T

            segments_tree_dict_pt_export_path = os.path.join(self.segments_dir, scene_name + self.segments_tree_dict_pt_suffix)
            torch.save(segments_tree_dict, segments_tree_dict_pt_export_path)
            print(f"[INFO] Segment tree saved as a torch dictionary to: {segments_tree_dict_pt_export_path}")

@hydra.main(config_path="../configs", config_name="search3d_scannet200_eval", version_base=None) 
def main(ctx: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    segment_computation = SegmentComputation(ctx=ctx)
    segment_computation.compute_segments_all()

if __name__ == "__main__":
    main()