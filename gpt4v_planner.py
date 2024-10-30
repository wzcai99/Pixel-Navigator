import numpy as np
from llm_utils.gpt_request import gptv_response
from llm_utils.nav_prompt import GPT4V_PROMPT
from cv_utils.detection_tools import *
from cv_utils.segmentation_tools import *
import cv2
import ast
class GPT4V_Planner:
    def __init__(self,dino_model,sam_model):
        self.gptv_trajectory = []
        self.dino_model = dino_model
        self.sam_model = sam_model
        self.detect_objects = ['bed','sofa','chair','plant','tv','toilet','floor']
    
    def reset(self,object_goal):
        # translation to align for the detection model
        if object_goal == 'tv_monitor':
            self.object_goal = 'tv'
        else:
            self.object_goal = object_goal

        self.gptv_trajectory = []
        self.panoramic_trajectory = []
        self.direction_image_trajectory = []
        self.direction_mask_trajectory = []

    def concat_panoramic(self,images,angles):
        try:
            height,width = images[0].shape[0],images[0].shape[1]
        except:
            height,width = 480,640
        background_image = np.zeros((2*height + 3*10, 3*width + 4*10,3),np.uint8)
        copy_images = np.array(images,dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
               continue
            copy_images[i] = cv2.putText(copy_images[i],"Angle %d"%angles[i],(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i//2) % 3
            background_image[10*(row+1)+row*height:10*(row+1)+row*height+height:,10*(col+1)+col*width:10*(col+1)+col*width+width,:] = copy_images[i]
        return background_image
    
    def make_plan(self,pano_images):
        direction,goal_flag = self.query_gpt4v(pano_images)
        direction_image = pano_images[direction]
        target_bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),self.detect_objects,self.dino_model)
        if self.detect_objects.index(self.object_goal) not in target_bbox.class_id:
            goal_flag = False

        if goal_flag:
            bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),[self.object_goal],self.dino_model)    
        else:
            bbox = openset_detection(cv2.cvtColor(direction_image,cv2.COLOR_BGR2RGB),['floor'],self.dino_model)
        try:
            mask = sam_masking(direction_image,bbox.xyxy,self.sam_model)
        except:
            mask = np.ones_like(direction_image).mean(axis=-1)
        
        self.direction_image_trajectory.append(direction_image)
        self.direction_mask_trajectory.append(mask)

        debug_image = np.array(direction_image)
        debug_mask = np.zeros_like(debug_image)
        pixel_y,pixel_x = np.where(mask>0)[0:2]
        pixel_y = int(pixel_y.mean())
        pixel_x = int(pixel_x.mean())
        debug_image = cv2.rectangle(debug_image,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,0,0),-1)
        debug_mask = cv2.rectangle(debug_mask,(pixel_x-8,pixel_y-8),(pixel_x+8,pixel_y+8),(255,255,255),-1)
        debug_mask = debug_mask.mean(axis=-1)
        return direction_image,debug_mask,debug_image,direction,goal_flag
        
    def query_gpt4v(self,pano_images):
        angles = (np.arange(len(pano_images))) * 30
        inference_image = cv2.cvtColor(self.concat_panoramic(pano_images,angles),cv2.COLOR_BGR2RGB)

        cv2.imwrite("monitor-panoramic.jpg",inference_image)
        text_content = "<Target Object>:{}\n".format(self.object_goal)
        self.gptv_trajectory.append("\nInput:\n%s \n"%text_content)
        self.panoramic_trajectory.append(inference_image)
        for i in range(10):
            try:
                raw_answer = gptv_response(text_content,inference_image,GPT4V_PROMPT)
                print("GPT-4V Output Response: %s"%raw_answer)
                answer = raw_answer
                answer = answer[answer.index("{"):answer.index("}")+1]
                answer = ast.literal_eval(answer)
                if 'Reason' in answer.keys() and 'Angle' in answer.keys():
                    break
                assert answer['Angle'] in angles
            except:
                continue
        self.gptv_trajectory.append("GPT-4V Answer:\n%s"%raw_answer)
        self.panoramic_trajectory.append(inference_image)
        try:
            return (int(answer['Angle']//30))%12,answer['Flag']
        except:
            return np.random.randint(0,12),False
