import torch
import cv2
import numpy as np
import quaternion
import time
from policy_network import PixelNav_Policy

class Policy_Agent:
    def __init__(self,model_path="./checkpoints/navigator.ckpt",max_token_length=64,image_size=224,device="cuda:0"):
        self.image_size = image_size
        self.max_token_length = max_token_length
        self.device = device
        self.network = PixelNav_Policy(max_token_length,device)
        self.network.load_state_dict(torch.load(model_path,map_location=device))
        self.network.eval()
    def reset(self,goal_image,goal_mask):
        self.history_image = np.zeros((self.max_token_length,self.image_size,self.image_size,3))
        self.goal_image = cv2.resize(cv2.cvtColor(goal_image,cv2.COLOR_BGR2RGB),(self.image_size,self.image_size))
        self.goal_image = self.goal_image[np.newaxis,:,:,:]
        self.goal_mask = cv2.resize(goal_mask,(self.image_size,self.image_size),cv2.INTER_NEAREST)
        self.goal_mask = self.goal_mask[np.newaxis,:,:,np.newaxis]
        self.predict_length = 0
        self.collide_times = 0
        self.collide_action = 0

    def step(self,obs_image,collide=False,early_stop=True):
        self.current_obs = cv2.resize(cv2.cvtColor(obs_image,cv2.COLOR_BGR2RGB),(self.image_size,self.image_size))
        self.history_image[self.predict_length] = self.current_obs.copy() #append(self.current_obs)
        self.input_image = self.history_image[np.newaxis,:,:,:,:]
        action_pred,distance_pred,goal_pred = self.network(self.goal_mask,self.goal_image,self.input_image)
        if collide == False:
            return_action = action_pred[0][self.predict_length].detach().cpu().numpy().argmax()
        else:
            action_pred[0][self.predict_length][1] = -1
            return_action = action_pred[0][self.predict_length].detach().cpu().numpy().argmax()
        return_distance = distance_pred[0][self.predict_length].detach().cpu().numpy()
        return_goal = np.array(obs_image)
        goal_mask = goal_pred[0][self.predict_length].detach().cpu().numpy()
        goal_y = goal_mask[0] * obs_image.shape[0]
        goal_x = goal_mask[1] * obs_image.shape[1]
        return_goal = cv2.rectangle(return_goal,(int(goal_x)-5,int(goal_y)-5),(int(goal_x)+5,int(goal_y)+5),(255,0,0),-1)
        return_goal = cv2.putText(return_goal,"%d"%int(return_distance[0]*10),(310,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6, cv2.LINE_AA)
        self.predict_length += 1
        if early_stop and self.predict_length > 32:
            return_action = 0
        return return_action,cv2.cvtColor(return_goal,cv2.COLOR_BGR2RGB)
        


