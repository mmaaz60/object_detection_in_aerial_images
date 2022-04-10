import argparse
import os
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-data_dir", "--dataset_dir", required=True,
                    help="Dataset base directory containing iSAID dataset. The dataset should be in COCO format.")
    ap.add_argument("-config", "--config_file", required=True,
                    help="Path to detectron2 config file for mask rcnn.")
    ap.add_argument("-output_dir", "--output_dir_path", required=False, default="./results",
                    help="Output dir path to save the checkpoints and progress of experiments. Default is './results'")
    ap.add_argument("-c", "--checkpoints", required=True, help="Path to the model checkpoints.")
    args = vars(ap.parse_args())

    return args


def register_datase(path):
    register_coco_instances("iSAID_val", {"thing_classes": ['ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
                                                            'basketball_court', 'Ground_Track_Field', 'Bridge',
                                                            'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
                                                            'Swimming_pool', 'Roundabout', 'Soccer_ball_field',
                                                            'plane', 'Harbor']},
                            f"{path}/val/instancesonly_filtered_val.json",
                            f"{path}/val/images/")


def prepare_config(config_path, **kwargs):
    # Parse the expected key-word arguments
    output_path = kwargs["output_dir"]
    checkpoints = kwargs["checkpoints"]

    # Create and initialize the config
    cfg = get_cfg()
    cfg.OUTPUT_DIR = output_path
    cfg.merge_from_file(config_path)
    cfg.DATASETS.TEST = ("iSAID_val",)
    cfg.MODEL.WEIGHTS = checkpoints
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    return cfg


def main():
    # Parse arguments
    args = parse_arguments()
    data_dir = args["dataset_dir"]
    config_file = args["config_file"]
    output_dir = args["output_dir_path"]
    output_path = f"{output_dir}/{os.path.basename(config_file).split('.')[0]}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    checkpoints = args["checkpoints"]

    # Setup logger
    setup_logger(f"{output_path}/log.txt")

    # Register dataset
    register_datase(data_dir)
    # Generate & save visualizations
    d2_config = prepare_config(config_file, output_dir=output_path, checkpoints=checkpoints)
    predictor = DefaultPredictor(d2_config)
    images_dir = f"{data_dir}/val/images"
    images_list = os.listdir(images_dir)
    for image in images_list:
        image_path = f"{images_dir}/{image}"
        img = cv2.imread(image_path)
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(d2_config.DATASETS.TEST[0]), scale=1.2)
        out_img = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_img_path = f"{output_path}/{image}"
        cv2.imwrite(output_img_path, out_img.get_image()[:, :, ::-1])


if __name__ == "__main__":
    main()
