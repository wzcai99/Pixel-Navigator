import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from constants import *
def initialize_sam_model(sam_encoder=SAM_ENCODER_VERSION,
                         sam_checkpoint=SAM_CHECKPOINT_PATH,
                         device="cuda:0"):
    sam = sam_model_registry[sam_encoder](checkpoint=sam_checkpoint).to(device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor
def sam_masking(image,bbox,sam_predictor):
    sam_predictor.set_image(image)
    masks,scores,_ = sam_predictor.predict(box=bbox,multimask_output=True)
    mask = masks[np.argmax(scores)]
    return mask