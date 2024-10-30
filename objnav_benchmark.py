import habitat
import os
import argparse
import csv
import cv2
import imageio
import numpy as np
from cv_utils.detection_tools import *
from tqdm import tqdm
from constants import *
from config_utils import hm3d_config
from gpt4v_planner import GPT4V_Planner
from policy_agent import Policy_Agent
from cv_utils.detection_tools import initialize_dino_model
from cv_utils.segmentation_tools import initialize_sam_model
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
def write_metrics(metrics,path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'],1024),cv2.COLOR_BGR2RGB)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_episodes",type=int,default=200)
    return parser.parse_known_args()[0]

def detect_mask(image,category,detect_model):
    det_result = openset_detection(image,category,detect_model)
    if det_result.xyxy.shape[0] > 0:
        goal_image = image
        goal_mask_xyxy = det_result.xyxy[np.argmax(det_result.confidence)]
        goal_mask_x = int((goal_mask_xyxy[0]+goal_mask_xyxy[2])/2)
        goal_mask_y = int((goal_mask_xyxy[1]+goal_mask_xyxy[3])/2)
        goal_mask = np.zeros((goal_image.shape[0],goal_image.shape[1]),np.uint8)
        goal_mask = cv2.rectangle(goal_mask,(goal_mask_x-8,goal_mask_y-8),(goal_mask_x+8,goal_mask_y+8),(255,255,255),-1)
        return True,goal_image,goal_mask
    return False,[],[]


args = get_args()
habitat_config = hm3d_config(stage='val',episodes=args.eval_episodes)
habitat_env = habitat.Env(habitat_config)
detection_model = initialize_dino_model()
segmentation_model = initialize_sam_model()

nav_planner = GPT4V_Planner(detection_model,segmentation_model)
nav_executor = Policy_Agent(model_path=POLICY_CHECKPOINT)
evaluation_metrics = []

for i in tqdm(range(args.eval_episodes)):
    find_goal = False
    obs = habitat_env.reset()
    dir = "./tmp/trajectory_%d"%i
    os.makedirs(dir,exist_ok=False)
    fps_writer = imageio.get_writer("%s/fps.mp4"%dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4"%dir,fps=4)
    heading_offset = 0

    nav_planner.reset(habitat_env.current_episode.object_category)
    episode_images = [obs['rgb']]
    episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]

    # a whole round planning process
    for _ in range(11):
        obs = habitat_env.step(3)
        episode_images.append(obs['rgb'])
        episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
    goal_image,goal_mask,debug_image,goal_rotate,goal_flag = nav_planner.make_plan(episode_images[-12:])    
    for j in range(min(11-goal_rotate,1+goal_rotate)):
        if goal_rotate <= 6:
            obs = habitat_env.step(3)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        else:
            obs = habitat_env.step(2)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
    nav_executor.reset(goal_image,goal_mask)


    while not habitat_env.episode_over:
        action,skill_image = nav_executor.step(obs['rgb'],habitat_env.sim.previous_step_collided)
        if action != 0 or goal_flag:
            if action == 4:
                heading_offset += 1
            elif action == 5:
                heading_offset -= 1
            obs = habitat_env.step(action)
            episode_images.append(obs['rgb'])
            episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
        else:
            if habitat_env.episode_over:
                break
            
            for _ in range(0,abs(heading_offset)):
                if habitat_env.episode_over:
                    break
                if heading_offset > 0:
                    obs = habitat_env.step(5)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    heading_offset -= 1
                elif heading_offset < 0:
                    obs = habitat_env.step(4)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                    heading_offset += 1
            
            # a whole round planning process
            for _ in range(11):
                if habitat_env.episode_over:
                    break
                obs = habitat_env.step(3)
                episode_images.append(obs['rgb'])
                episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            goal_image,goal_mask,debug_image,goal_rotate,goal_flag = nav_planner.make_plan(episode_images[-12:])
            for j in range(min(11-goal_rotate,goal_rotate+1)):
                if habitat_env.episode_over:
                    break
                if goal_rotate <= 6:
                    obs = habitat_env.step(3)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
                else:
                    obs = habitat_env.step(2)
                    episode_images.append(obs['rgb'])
                    episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
            nav_executor.reset(goal_image,goal_mask)
    
    for image,topdown in zip(episode_images,episode_topdowns):
        fps_writer.append_data(image)
        topdown_writer.append_data(topdown)
    fps_writer.close()
    topdown_writer.close()

    evaluation_metrics.append({'success':habitat_env.get_metrics()['success'],
                               'spl':habitat_env.get_metrics()['spl'],
                               'distance_to_goal':habitat_env.get_metrics()['distance_to_goal'],
                               'object_goal':habitat_env.current_episode.object_category})
    write_metrics(evaluation_metrics)
    

    

            

        

        


    
        


