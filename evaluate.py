from typing import Tuple
try:
    import faster_coco_eval
    # Replace pycocotools with faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
except ImportError:
    print("faster_coco_eval not found, using pycocotools instead.")
    faster_coco_eval = None
    import pycocotools

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_GT_PATH = 'data/coco2017/instances_val2017.json'

def evaluate_coco(
    pd_path: str,
    gt_path: str = DEFAULT_GT_PATH,
    iou_type: str = 'bbox',
) -> Tuple[float, float]:
    anno = COCO(gt_path)
    try:
        pred = anno.loadRes(pd_path)
    except IndexError as e:
        print(f"No label found in {pd_path}")
        return .0, .0
    
    coco_eval = COCOeval(anno, pred, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[:2]


if __name__ == "__main__":
    """
    Usage:
        python evaluate.py results.json [instances_val2017.json]
        
    Arguments:
        results.json: Path to the JSON file containing the predictions.
        ground_truth_file_path (optional): Path to the ground truth annotations file.
            Defaults to 'data/coco2017/instances_val2017.json'.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("gt", nargs="?", type=str, default="data/coco2017/instances_val2017.json")
    args = parser.parse_args()

    evaluate_coco(args.path, args.gt)
