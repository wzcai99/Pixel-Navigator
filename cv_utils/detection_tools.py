from groundingdino.util.inference import Model
from constants import *
import torchvision
import torch
def initialize_dino_model(dino_config=GROUNDING_DINO_CONFIG_PATH,
                          dino_checkpoint=GROUNDING_DINO_CHECKPOINT_PATH,
                          device="cuda:0"):
    model = Model(model_config_path=dino_config,
                  model_checkpoint_path=dino_checkpoint,
                  device=device)
    return model

def openset_detection(image,target_classes,dino_model,
                      box_threshold=0.2,
                      text_threshold=0.4,
                      nms_threshold=0.5):
    detections = dino_model.predict_with_classes(image=image,
                                                classes=target_classes,
                                                box_threshold=box_threshold,
                                                text_threshold=text_threshold)
    detections.xyxy = detections.xyxy[detections.class_id!=None]
    detections.confidence = detections.confidence[detections.class_id!=None]
    detections.class_id = detections.class_id[detections.class_id!=None]
    nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy), 
                                torch.from_numpy(detections.confidence), 
                                nms_threshold).numpy().tolist()
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    return detections