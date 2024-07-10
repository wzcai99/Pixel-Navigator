# data-collection related directory
HABITAT_ROOT_DIR = "{YOUR_HABITAT_DIR}"
HM3D_CONFIG_PATH = f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
MP3D_CONFIG_PATH = f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml"
SCENE_PREFIX = "{YOUR_SCENE_DIR}"
EPISODE_PREFIX = "{YOUR_DATASET_DIR}"
# detection & segmentation related configs and checkpoints
GROUNDING_DINO_CONFIG_PATH = "./checkpoints/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoints/groundingdino_swinb_cogcoor.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./checkpoints/sam_vit_h_4b8939.pth"
# policy checkpoint
POLICY_CHECKPOINT = "./checkpoints/navigator.pth

