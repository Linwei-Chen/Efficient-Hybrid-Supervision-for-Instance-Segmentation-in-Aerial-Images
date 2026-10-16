import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,6,7'
print(os.environ['CUDA_VISIBLE_DEVICES'])

import torch

print(torch.cuda.device_count())

import torch.distributed as dist

import numpy as np
import cv2
import random
import os.path as osp

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, launch, DefaultTrainer
# from detectron2.config import get_cfg
from adet.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine.defaults import default_argument_parser


def predict_test():
    im = cv2.imread("/home/charles/Documents/test.png")
    cv2.imshow('temp', im)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    print(outputs)
    # outputs["instances"].pred_classes
    # outputs["instances"].pred_boxe
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('result', v.get_image()[:, :, ::-1])
    cv2.waitKey()


def model_build_test():
    im = cv2.imread("/home/charles/Documents/test.png")
    cv2.imshow('temp', im)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    model = build_model(cfg)
    print(model)

    cfg.MODEL.WEIGHTS = '~/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('result', v.get_image()[:, :, ::-1])
    cv2.waitKey()

    # DetectionCheckpointer(model, osp.expanduser('~/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth'))

    pass


def train(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()


# prepare dataset
# 1. train set
iSAID_path = '~/dataset/iSAID/'
train_img_path = osp.join(iSAID_path, './iSAID_patches/train/images')
train_json_path = osp.join(iSAID_path, './iSAID_patches/train/instancesonly_filtered_train.json')
register_coco_instances("iSAID_train", {},
                        json_file=osp.expanduser(train_json_path),
                        image_root=osp.expanduser(train_img_path))
# iSAID_train_metadata = MetadataCatalog.get("iSAID_train")
# iSAID_train_dicts = DatasetCatalog.get("iSAID_train")

# val set
val_img_path = osp.join(iSAID_path, './iSAID_patches/val/images')
val_json_path = osp.join(iSAID_path, './iSAID_patches/val/instancesonly_filtered_val.json')
register_coco_instances("iSAID_val", {},
                        json_file=osp.expanduser(val_json_path),
                        image_root=osp.expanduser(val_img_path))


# iSAID_val_metadata = MetadataCatalog.get("iSAID_val")
# iSAID_val_dicts = DatasetCatalog.get("iSAID_val")

def train_and_eval():
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.merge_from_file('FCOS-Detection/MS_R_101_2x.yaml')
    cfg.OUTPUT_DIR = 'output/fcos/MS_R_101_2x_deform'

    cfg.DATASETS.TRAIN = ("iSAID_train",)
    cfg.DATASETS.TEST = ("iSAID_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    # Default
    # MODEL:
    #     WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 180000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    # cfg.SOLVER.STEPS = (60000, 120000)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.FCOS.NUM_CLASSES = 15  # iSAID
    cfg.MODEL.FCOS.USE_DEFORMABLE = True
    cfg.VIS_PERIOD = 10
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    args = default_argument_parser().parse_args()
    args.num_gpus = torch.cuda.device_count()
    print("Command Line Args:", args)
    launch(
        train,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(cfg,),
    )
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)

    evaluator = COCOEvaluator("iSAID_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "iSAID_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == '__main__':
    # model_build_test()
    train_and_eval()
    pass
