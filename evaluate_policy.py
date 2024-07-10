import habitat
import json
import os
import argparse
import cv2
import quaternion
import imageio
import shutil
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from policy_agent import Policy_Agent
from data_utils.geometry_tools import *
from config_utils import *
from constants import *

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def visualize_target(rgb,mask):
    copy_rgb = rgb.copy()
    copy_rgb[mask!=0] = np.array([0,0,255])
    return copy_rgb

def random_pixel_goal(habitat_config,habitat_env, difficulty='medium'):
    camera_int = habitat_camera_intrinsic(habitat_config)
    robot_pos = habitat_env.sim.get_agent_state().position
    robot_rot = habitat_env.sim.get_agent_state().rotation
    camera_pos = habitat_env.sim.get_agent_state().sensor_states['rgb'].position
    camera_rot = habitat_env.sim.get_agent_state().sensor_states['rgb'].rotation
    camera_obs = habitat_env.sim.get_observations_at(robot_pos,robot_rot)
    rgb = camera_obs['rgb']
    depth = camera_obs['depth']
    xs,zs,rgb_points,rgb_colors = get_pointcloud_from_depth(rgb,depth,camera_int)
    rgb_points = translate_to_world(rgb_points,camera_pos,quaternion.as_rotation_matrix(camera_rot))
    if difficulty == 'easy':
        condition_index = np.where((rgb_points[:,1] < robot_pos[1] + 1.0) & (rgb_points[:,1] > robot_pos[1] - 0.2) & (depth[(zs,xs)][:,0] > 1.0) & (depth[(zs,xs)][:,0] < 3.0))[0]
    elif difficulty == 'medium':
        condition_index = np.where((rgb_points[:,1] < robot_pos[1] + 1.0) & (rgb_points[:,1] > robot_pos[1] - 0.2) & (depth[(zs,xs)][:,0] > 3.0) & (depth[(zs,xs)][:,0] < 5.0))[0]
    else:
        raise NotImplementedError
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(rgb_points[condition_index])
    pointcloud.colors = o3d.utility.Vector3dVector(rgb_colors[condition_index]/255.0)
    if condition_index.shape[0] == 0:
        return False,[],[],[],[]
    else:
        random_index = np.random.choice(condition_index)
        target_x = xs[random_index]
        target_z = zs[random_index]
        target_point = rgb_points[random_index]
        min_z = max(target_z-5,0)
        max_z = min(target_z+5,depth.shape[0])
        min_x = max(target_x-5,0)
        max_x = min(target_x+5,depth.shape[1])
        target_mask = np.zeros((depth.shape[0],depth.shape[1]),np.uint8)
        target_mask[min_z:max_z,min_x:max_x] = 255
        target_point[1] = robot_pos[1]
        geodesic_distance = habitat_env.sim.geodesic_distance(target_point,robot_pos)
        return True,np.array(rgb),target_mask,target_point,geodesic_distance

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",type=str,choices=['hm3d','mp3d'],default='mp3d')
    parser.add_argument("--difficulty",type=str,choices=['easy','medium'],default='easy')
    parser.add_argument("--max_length",type=int,default=64)
    parser.add_argument("--episodes",type=int,default=1000)
    parser.add_argument("--robot_height",type=float,default=0.88)
    parser.add_argument("--robot_radius",type=float,default=0.18)
    parser.add_argument("--sensor_height",type=float,default=0.88)
    parser.add_argument("--step_size",type=float,default=0.25)
    parser.add_argument("--turn_angle",type=int,default=30)
    parser.add_argument("--image_width",type=int,default=640)
    parser.add_argument("--image_height",type=int,default=480)
    parser.add_argument("--image_hfov",type=int,default=79)
    args = parser.parse_known_args()[0]
    return args

args = get_args()
if args.prefix == 'mp3d':
    habitat_config = mp3d_data_config(stage='val',
                                 episodes=args.episodes,
                                 robot_height=args.robot_height,
                                 robot_radius=args.robot_radius,
                                 sensor_height=args.sensor_height,
                                 image_width=args.image_width,
                                 image_height=args.image_height,
                                 image_hfov=args.image_hfov,
                                 step_size=args.step_size,
                                 turn_angle=args.turn_angle)
elif args.prefix == 'hm3d':
    habitat_config = hm3d_data_config(stage='val',
                                 episodes=args.episodes,
                                 robot_height=args.robot_height,
                                 robot_radius=args.robot_radius,
                                 sensor_height=args.sensor_height,
                                 image_width=args.image_width,
                                 image_height=args.image_height,
                                 image_hfov=args.image_hfov,
                                 step_size=args.step_size,
                                 turn_angle=args.turn_angle)

env = habitat.Env(habitat_config)
policy_agent = Policy_Agent(model_path=POLICY_CHECKPOINT)
oracle_agent = ShortestPathFollower(env.sim,0.5,False)
metrics_sr = {'easy':[],'medium':[],'hard':[]}
metrics_spl = {'easy':[],'medium':[],'hard':[]}

for i in tqdm(range(args.episodes)):
    os.makedirs("%s_eval_trajectory/evaluate_%s_%d/"%(args.prefix,args.prefix,i),exist_ok=True)
    timesteps = 0
    obs = env.reset()
    goal_flag,goal_image,goal_mask,goal_point,goal_dist = random_pixel_goal(habitat_config,env,args.difficulty)
    if goal_flag == False or oracle_agent.get_next_action(goal_point) == 0:
        shutil.rmtree("%s_eval_trajectory/evaluate_%s_%d/"%(args.prefix,args.prefix,i))
        continue
    policy_agent.reset(goal_image,goal_mask)
    image_writer = imageio.get_writer("%s_eval_trajectory/evaluate_%s_%d/"%(args.prefix,args.prefix,i)+"fps.mp4", fps=4)

    move_distance = 0
    last_position = env.sim.get_agent_state().position
    while True:
        if env.episode_over or timesteps >= args.max_length:
            break          
        image = obs['rgb']
        collide = env.sim.previous_step_collided
        pred_action,pred_mask = policy_agent.step(image,collide)
        obs = env.step(pred_action)
        move_distance += np.sqrt(np.sum(np.square(last_position - env.sim.get_agent_state().position)))
        last_position = env.sim.get_agent_state().position
        timesteps += 1
        concat_image = np.concatenate((visualize_target(goal_image,goal_mask),cv2.cvtColor(pred_mask,cv2.COLOR_BGR2RGB)),axis=1)
        image_writer.append_data(concat_image)
        
    if goal_dist < 3.0:
        sr = (np.sqrt(np.sum(np.square(goal_point - env.sim.get_agent_state().position))) < 1.0)
        spl = np.clip(sr * goal_dist / (move_distance+1e-6),0,1)
        metrics_sr['easy'].append(sr)
        metrics_spl['easy'].append(spl)
    elif goal_dist > 3.0 and goal_dist < 5.0:
        sr = (np.sqrt(np.sum(np.square(goal_point - env.sim.get_agent_state().position))) < 1.0)
        spl = np.clip(sr * goal_dist / (move_distance+1e-6),0,1)
        metrics_sr['medium'].append(sr)
        metrics_spl['medium'].append(spl)
    
    print('easy episode = %d'%len(metrics_sr['easy']))
    print('easy sr = %.4f'%np.array(metrics_sr['easy']).mean())
    print('easy spl = %.4f'%np.array(metrics_spl['easy']).mean())

    print('medium episode = %d'%len(metrics_sr['medium']))
    print('medium sr = %.4f'%np.array(metrics_sr['medium']).mean())
    print('medium spl = %.4f'%np.array(metrics_spl['medium']).mean())
    image_writer.close()


       
    

