# Search3D ScanNet++ benchmark data: open-vocabulary part annotations on a small subset of the ScanNet++ dataset

This release provides the benchmark data consisting of open-vocabulary part annotations on a small subset of the ScanNet++ dataset to support open-vocabulary 3D part segmentation evaluation, as introduced in our IEEE RA-L 2025 paper *Search3D: Hierarchical Open-Vocabulary 3D Segmentation*. This evaluation set provides annotations on laser scans from a subset of the ScanNet++ dataset. More specifically, our small evaluation set includes 14 object and 20 part annotations across 8 ScanNet++ scenes, along with open-vocabulary text descriptions.

> **Download:**  
> You can download the dataset annotations for the ScanNet++ Search3D evaluation set from [here](https://drive.google.com/drive/folders/1mKNSG_HTEzwT18lTtfGVM7uq76K44kzA?usp=sharing). Next, please unpack the data following the structure described in the following section. 


- **Object Annotations:** Object-level annotations in the annotation hierarchy.
- **Open-Vocabulary Part (OV-Part) Annotations:** For OV part annotations, we use joint object-part name tuples that includes both object and part labels (e.g., "desk drawer", "cabinet door"). **This is the primary benchmark data we used to report additional part segmentation results in our Search3D paper.**

---

## Dataset Structure

The dataset is organized into the following folders:
```
  scannetpp_evaluation_data_search3d/
    ├──scannetpp_data_search3d/ # all annotations per-point for each scene. 
    │    ├── obj_annotations
    │    │   ├── <XXXXXXXXXX>_<OBJECT_ANNOTATION_ID>.txt # first part is the scene ID from ScanNet v1
    │    │   ├── ... (other scene files)
    │    │   └── obj_query_dict.json # json file describing the query text corresponding to each object
    │    └── ov_part_annotations
    │        ├── <XXXXXXXXXX>_<OBJECT_ANNOTATION_ID>_<PART_ANNOTATION_ID>.txt # first part is the scene ID from ScanNet v1, next part is the object annotation ID (same as the ID in the obj_annotations folder), then finally the part annotation ID.
    │        ├── ... (other scene files)
    │    │   └── ov_part_query_dict.json # json file describing the query text corresponding to each part annotation
    └──scannetpp_data/ # ply files for each scene, can point at the original ScanNet++ data location.
         ├── <XXXXXXXXXX>/scans/mesh_aligned_0.05.ply
         └── ... (other scene folders)
```
For example, the part annotation ID `25f3b7a318_40bc2005-5121-489f-9f61-21795fce1e9e_2535632e-efab-49d8-b232-7c87bf09e677.txt` means that the scene ID is `25f3b7a318`, object annotation ID is `40bc2005-5121-489f-9f61-21795fce1e9e` and the annotation ID of the part within this object is `2535632e-efab-49d8-b232-7c87bf09e677`. The `obj_query_dict.json` and `ov_part_query_dict.json` files can be used to identify the query texts corresponding to each annotation.

### Annotation Format in **`scannetpp_data_search3d`**
- **Per-point labeling:**  
  Each line in an annotation file represents a label for a corresponding point in the original `.ply` file (point cloud), which can be found under the paths **`<SCENE_ID>/scans/mesh_aligned_0.05.ply`** in the original ScanNet++ dataset.
  
- **`obj_annotations`:**  
  Contains per-scene annotation files for **object instance segmentation** that correspond to the query specified in `obj_query_dict.json`. Each file (e.g., `<XXXXXXXXXX>_<OBJECT_ANNOTATION_ID>.txt`) provides a label per point corresponding to the object instances in the scene that correspond to the specified query text. Label 0 is used for any background point, and labels 1001, 1002, 1003, ... are used to identify each individual instance ID that correspond to separate each object, e.g. different chairs in a room. Please note that in most scenes there are only a single instance that correspond to the query, e.g. labels 0 and 1001 are the only labels.

- **`ov_part_annotations`:**  
  Contains per-scene annotation files for **open-vocabulary part segmentation** that correspond to the query specified in `ov_part_query_dict.json`. Each file (e.g., `<XXXXXXXXXX>_<OBJECT_ANNOTATION_ID>_<PART_ANNOTATION_ID>.txt`) provides a label per point corresponding to the part instances in the scene that correspond to the specified query text. Label 0 is used for any background point, and labels 1001, 1002, 1003, ... are used to identify each individual instance ID that correspond to separate parts, e.g. different legs of a chair.


## Evaluation


> **Important Note:**  
> For all evaluations related to open-vocabulary 3D part segmentation, **we only use the `ov_part_annotations` data.**. All experiments and benchmarking reported in our paper are based on the data in the **`ov_part_annotations`** folder. However, this folder includes annotations for more scenes than used in the final evaluation. A list of test scenes used in our benchmark is provided here: [benchmark/evaluation/scannetpp_subset/data/test_scenes_scannetpp_subset.txt](https://github.com/aycatakmaz/search3d/benchmark/evaluation/scannetpp_subset/data/test_scenes_scannetpp_subset.txt)

---

## Licensing

This dataset is released under the original ScanNet++ dataset license. All terms and conditions specified in the [ScanNet++ License Information](https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf) apply to this dataset. Please ensure that you comply with these terms when using the data in your research. Due to the conditions of this term, we are unable to share the original PLY files or any other assets corresponding to the scenes we used in our small benchmark data. Please follow the original instructions to obtain access to the ScanNet++ scans and data. The 8 scenes required for evaluation are listed here: [benchmark/evaluation/scannetpp_subset/data/test_scenes_scannetpp_subset.txt](https://github.com/aycatakmaz/search3d/benchmark/evaluation/scannetpp_subset/data/test_scenes_scannetpp_subset.txt).

---

## Citation

If you use this small benchmark in your research, we would really appreciate if you could please cite our paper:

```bibtex
@article{takmaz2025search3d,
  title={{Search3D: Hierarchical Open-Vocabulary 3D Segmentation}},
  author={Takmaz, Ayca and Delitzas, Alexandros and Sumner, Robert W. and Engelmann, Francis and Wald, Johanna and Tombari, Federico},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2025}
}
```

and the original ScanNet++ dataset:

```bibtex
@inproceedings{yeshwanth2023scannet++,
  title={Scannet++: A high-fidelity dataset of 3d indoor scenes},
  author={Yeshwanth, Chandan and Liu, Yueh-Cheng and Nie{\ss}ner, Matthias and Dai, Angela},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12--22},
  year={2023}
}
```

