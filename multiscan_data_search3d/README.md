# Search3D Data Release: An Adaptation of MultiScan for Hierarchical Open-Vocabulary 3D Segmentation

This release provides an adapted version of the original MultiScan dataset to support open-vocabulary 3D part segmentation evaluation, as introduced in our RA-L 2025 paper *Search3D: Hierarchical Open-Vocabulary 3D Segmentation*. 


> **Download:**  
> You can download the adapted version of the MultiScan evaluation dataset from [here](https://drive.google.com/drive/folders/1pyTHX3Ym8-StdWDvCVv72C6N_8GsjvSx?usp=sharing). Next, please unpack the data following the structure described in the following section. 


The original MultiScan dataset was designed with fine-grained part annotations for articulated part segmentation. However, to evaluate open-vocabulary segmentation across different granularity levels and capture the scene-object-part hierarchy, we reanalyzed the dataset and formed three distinct groups based on existing annotations:

- **Object Annotations:** For scene-scale object instance segmentation.  
- **Part Annotations:** For part segmentation with only part names (e.g., "drawer", "door")—lacking object context.  
- **Open-Vocabulary Part (OV-Part) Annotations:** For OV part annotations, we use joint object-part name tuples that includes both object and part labels (e.g., "desk drawer", "cabinet door"). **This is the primary benchmark data for our work.**

---

## Dataset Structure

The dataset is organized into the following folders:
```
  multiscan_evaluation_data_search3d/
    ├──multiscan_annotations_search3d/ # all annotations per-point for each scene. 
    │    ├── obj_annotations
    │    │   ├── scene_000XX_XX_obj_inst.txt
    │    │   └── ... (other scene files)
    │    ├── part_annotations
    │    │   ├── scene_000XX_XX_part_inst.txt
    │    │   └── ... (other scene files)
    │    └── ov_part_annotations
    │        ├── scene_000XX_XX_obj_part_inst.txt
    │        └── ... (other scene files)
    ├──multiscan_test_plys_only/ # ply files for each scene
    │    ├── scene_000XX_XX.ply
    │    ├── scene_000XX_XX.ply
    │    └── ... (other scene files)
    └──multiscan_search3d_constants.py # file with text descriptions for each semantic label, for object, part and OV-part segmentation.
  ```


### Details for the annotation folder **`multiscan_annotations_search3d`**
- **`obj_annotations`:**  
  Contains per-scene annotation files for **object instance segmentation**. Each file (e.g., `scene_000XX_XX_obj_inst.txt`) provides a label per point corresponding to different object instances in the scene.

- **`part_annotations`:**  
  Contains per-scene annotation files for **part segmentation**. Each file (e.g., `scene_000XX_XX_part_inst.txt`) assigns part labels to points without specifying the associated object. For instance, a "drawer" could belong to a cupboard or a desk, but the annotation simply uses "drawer" for both.

- **`ov_part_annotations`:**  
  Contains per-scene annotation files for **open-vocabulary part segmentation**, where parts are annotated jointly with their objects (e.g., `scene_000XX_XX_obj_part_inst.txt`). These files include more semantically meaningful composite labels (such as "desk drawer" or "cabinet door") that resolve potential ambiguities inherent in the standalone part annotations.

---

## Annotation Format for OV-Parts (Joint Obj-Part Annotations)

Each annotation file in the dataset corresponds to a single scene and is structured as follows:

- **Per-Point Labeling:**  
  Each line in an annotation file represents a label for a corresponding point in the original `.ply` file (point cloud), which can be found under the **`multiscan_test_plys_only`** folder with all scene mesh files.  
  - A label of `0` indicates that the point is unannotated (background).  
  - Non-zero labels are composite numerical IDs encoding both the semantic and instance information.

- **Composite ID Format (`AAAIII`):**  
  - The **first part (`AAA`)** represents the object-part semantic ID (e.g., `121003` might correspond to "dishwasher drawer").  
  - The **second part (`III`)** indicates the instance number for that semantic class (e.g., `002` for the second instance).

**Example:**  
For a label `121003002`:  
- **Semantic ID:** `121003` → This maps to a joint object-part category (e.g., "dishwasher drawer").  
- **Instance ID:** `002` → This is the second occurrence of "dishwasher drawer" in the scene.

This composite labeling ensures that each annotated point carries both semantic meaning and instance-specific identification, which is critical for fine-grained segmentation tasks.

---

## Data Constants and Label Mappings

The repository includes a Python file (e.g., `multiscan_search3d_constants.py`) that defines the following constants:

- **Object Class Labels:**  
  `CLASS_LABELS` — A tuple of strings specifying object category names used in object instance segmentation.

- **Part Class Labels:**  
  `CLASS_LABELS_PARTS` — A tuple of strings listing the part category names used in the standalone part annotations.

- **Valid Class IDs:**  
  `VALID_CLASS_IDS` and `VALID_CLASS_IDS_PARTS` — Tuples of integer IDs corresponding to valid object and part categories, respectively.

- **Joint Annotation Details:**  
  - `VALID_JOINT_TUPLE_NAMES`: A list of tuples representing valid (object, part) pairs (e.g., `('door', 'frame')`).  
  - `VALID_JOINT_TUPLE_IDS`: Numeric tuples that map to the valid object-part pairs.  
  - `VALID_JOINT_SEMANTIC_IDS`: Composite IDs used in the annotations that uniquely identify joint object-part categories.  
  - `VALID_JOINT_LABEL_LIST`: A list of descriptive string labels (e.g., `"door frame"`, `"desk drawer"`) for the joint categories.

- **Full Object-Part Tuple Set:**  
  `obj_part_label_set_joint_TUPLES_FULL` — A comprehensive list of joint object-part labels, including extended or more descriptive pairings (e.g., `('cabinet', 'countertop')`).

---

## Evaluation


> **Important Reminder:**  
> For all evaluations related to open-vocabulary 3D part segmentation, **we only use the `ov_part_annotations` data.** The annotations in `part_annotations` include only part labels without object context, which can lead to ambiguity (please see our paper for further detail). All experiments and benchmarking reported in our paper are based on the data in the **`ov_part_annotations`** folder. 

---

## Licensing

This dataset is released under the original MultiScan dataset license. All terms and conditions specified in the [MultiScan License Information](https://3dlg-hcvc.github.io/multiscan/#) apply to this dataset. Please ensure that you comply with these terms when using the data in your research.

---

## Citation

If you use this dataset in your research, we would really appreciate if you could please cite our paper:

```bibtex
@article{takmaz2025search3d,
  title={{Search3D: Hierarchical Open-Vocabulary 3D Segmentation}},
  author={Takmaz, Ayca and Delitzas, Alexandros and Sumner, Robert W. and Engelmann, Francis and Wald, Johanna and Tombari, Federico},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2025}
}
```

and the original MultiScan dataset:

```bibtex
@inproceedings{mao2022multiscan,
  title={MultiScan: Scalable RGBD scanning for 3D environments with articulated objects},
  author={Mao, Yongsen and Zhang, Yiming and Jiang, Hanxiao and Chang, Angel X and Savva, Manolis},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2022}
}
```

