import os
import numpy as np
#from benchmark.evaluation.scannetpp_subset.eval_semantic_instance_parts_OV import evaluate
from eval_semantic_instance_parts_OV import evaluate
import tqdm
import json
import pdb
import torch

from tqdm import tqdm


def test_pipeline_scannetpp_subset(pred_dir = "PATH/TO/PREDICTIONS",
                                    gt_dir = "PATH/TO/GT/ANNOTATIONS",
                                    dataset_type='scannetpp_search3d_ov_parts', 
                                    query_dict_path = "benchmark/evaluation/scannetpp_subset/data/ov_part_query_dict.json",
                                    test_scenes_list="benchmark/evaluation/scannetpp_subset/data/test_scenes_scannetpp_subset.txt"):

    scene_names = [el.strip() for el in open(test_scenes_list).readlines()]
    query_dict = json.load(open(query_dict_path))

    preds = {}
    for scene_name in tqdm(scene_names):
        # load query texts in the query_dict for this scene
        scene_query_dict = {key: value for key, value in query_dict.items() if key.startswith(scene_name)}
        # scene_query_dict is a dict whose keys are annotation IDs for the annotations that correspond to the scene <scene_name>
        # e.g. for scene 25f3b7a318, we have two annotations, one is annot_id='25f3b7a318_40bc2005-5121-489f-9f61-21795fce1e9e_e31b2f88-7d27-4ddf-92b5-ba7b3778be10'
        # scene_query_dict[annot_id] is a list of a query text pair for this annotation, namely ["laptop", "keyboard"] where laptop is the object name and the keyboard is the part name

        raise NotImplementedError("Please adapt the code below to load your own predictions for each query text")
        # -- ADAPT THE PART BELOW --
        pred_dict = torch.load(os.path.join(pred_dir, '{}_res_dict.pt'.format(scene_name)))
        res = process_predictions_and_get_segmentation_masks_per_query(pred_dict, scene_query_dict) # replace this with your own predictions
        # -- ADAPT THE PART ABOVE --

        for annot_id in res.keys():
            preds[annot_id] = {
                'pred_masks': res[annot_id]['pred_masks'],
                'pred_scores': res[annot_id]['pred_scores'], #confidence scores for each mask
                'pred_classes': np.ones_like(res[annot_id]['pred_scores']) } # class labels for each mask, in this case all ones as we each annotation as "target" class

    inst_AP = evaluate(preds, gt_dir, dataset=dataset_type)


if __name__ == '__main__':

    test_pipeline_scannetpp_subset(pred_dir = "/Users/aycatakmaz/search3d/gt_dir_test", #"PATH/TO/PREDICTIONS",
                                    gt_dir = "/Users/aycatakmaz/search3d/scannetpp_evaluation_data_search3d/scannetpp_annotations_search3d/ov_part_annotations", #"PATH/TO/GT/ANNOTATIONS/scannetpp_annotations_search3d/ov_part_annotations",
                                    dataset_type="scannetpp_search3d_ov_parts", 
                                    query_dict_path = "/Users/aycatakmaz/search3d/benchmark/evaluation/scannetpp_subset/data/ov_part_query_dict.json",
                                    test_scenes_list="/Users/aycatakmaz/search3d/benchmark/evaluation/scannetpp_subset/data/test_scenes_scannetpp_subset.txt")