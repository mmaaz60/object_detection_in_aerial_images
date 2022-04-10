import argparse
import os
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (
            COCOEvaluator, DatasetEvaluators
            )
from detectron2.checkpoint import DetectionCheckpointer
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
    ap.add_argument("--resume", action='store_true', help="Either to resume training or not.")
    ap.add_argument("--eval_only", action='store_true', help="Either to perform only the evaluation.")
    ap.add_argument("--eval_checkpoints", required=False, help="Checkpoints path to load weights from for evaluation.")
    args = vars(ap.parse_args())

    return args


def register_datase(path):
    register_coco_instances("iSAID_train", {},
                            f"{path}/train/instancesonly_filtered_train.json",
                            f"{path}/train/images/")
    register_coco_instances("iSAID_val", {},
                            f"{path}/val/instancesonly_filtered_val.json",
                            f"{path}/val/images/")


def prepare_config(config_path, **kwargs):
    # Parse the expected key-word arguments
    output_path = kwargs["output_dir"]
    workers = kwargs["workers"]

    # Create and initialize the config
    cfg = get_cfg()
    cfg.SEED = 26911042  # Fix the random seed to improve consistency across different runs
    cfg.OUTPUT_DIR = output_path
    cfg.merge_from_file(config_path)
    cfg.DATASETS.TRAIN = ("iSAID_train",)
    cfg.DATASETS.TEST = ("iSAID_val",)
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
    # Training schedule - equivalent to 0.5x schedule as per Detectron2 criteria
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.STEPS = (60000, 80000)
    cfg.SOLVER.MAX_ITER = 90000

    return cfg


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return DatasetEvaluators([COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)])


def main():
    # Parse arguments
    args = parse_arguments()
    data_dir = args["dataset_dir"]
    config_file = args["config_file"]
    output_dir = args["output_dir_path"]
    output_path = f"{output_dir}/{os.path.basename(config_file).split('.')[0]}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    resume = args["resume"]
    eval_only = args["eval_only"]
    eval_checkpoints = args["eval_checkpoints"]
    if eval_only:
        assert eval_checkpoints is not None

    # Setup logger
    setup_logger(f"{output_path}/log.txt")

    # Register dataset
    register_datase(data_dir)
    # Prepare configuration
    d2_config = prepare_config(config_file, output_dir=output_path, workers=5)
    # Evaluation only using the provided checkpoints path
    if eval_only:
        d2_config.MODEL.WEIGHTS = eval_checkpoints
        model = Trainer.build_model(d2_config)
        DetectionCheckpointer(model, save_dir=d2_config.OUTPUT_DIR).load(eval_checkpoints)
        Trainer.test(d2_config, model)
        return
    # Training and evaluation
    trainer = Trainer(d2_config)
    trainer.resume_or_load(resume=resume)
    trainer.train()


if __name__ == "__main__":
    main()
