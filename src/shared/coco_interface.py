from pycocotools.coco import COCO
from pathlib import Path
import os


def get_coco_classes(coco_configs_path: Path, species: str):
    coco_json = os.path.join(coco_configs_path, species.strip()+"_annotations.coco.json")

    coco_valid = COCO(coco_json)
    class_to_name = dict()
    for a_dict in coco_valid.loadCats(coco_valid.getCatIds()):
        class_to_name[a_dict["id"]] = a_dict["name"]
    return class_to_name