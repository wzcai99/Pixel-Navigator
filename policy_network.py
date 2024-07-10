import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18 as imagenet_resnet18

# generate the trajectory mask for self-attention
def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PixelNav_Policy(nn.Module):
    def __init__(self,max_token_length=64,device='cuda:0'):
        super().__init__()
        self.device = device
        self.max_token_length = max_token_length
        # resnet backbone use to encode all the history RGB images, randomly initialized
        self.history_backbone = nn.Sequential(*(list(imagenet_resnet18().children())[:-1]),nn.Flatten()).to(device)
        # goal encoder to encode both the initial RGB image and the goal mask, 4-channel input
        self.goal_backbone = imagenet_resnet18()
        self.goal_backbone.conv1 = nn.Conv2d(4,self.goal_backbone.conv1.out_channels,
                                             kernel_size=self.goal_backbone.conv1.kernel_size,
                                             stride=self.goal_backbone.conv1.stride,
                                             padding=self.goal_backbone.conv1.padding,
                                             bias=self.goal_backbone.conv1.bias)
        self.goal_backbone = nn.Sequential(*(list(self.goal_backbone.children())[:-1]),nn.Flatten()).to(device)
        # goal fusion, project the representations to all the input tokens
        self.goal_concat_proj = nn.Linear(512,256,device=device)
        # goal input token
        self.goal_input_proj = nn.Linear(512,768,device=device)
        # transformer-decoder policy
        self.dt_policy = nn.TransformerDecoder(nn.TransformerDecoderLayer(768,4,dropout=0.25,batch_first=True,device=device),4)
        self.po_embedding = nn.Embedding(max_token_length,768,device=device)
        nn.init.normal_(self.po_embedding.weight,0,0.01)
        # prediction heads, including policy head, tracking head and distance head
        self.action_head = nn.Linear(768,6,device=device)
        self.distance_head = nn.Linear(768,1,device=device)
        self.goal_head = nn.Linear(768,2,device=device)
    
    def forward(self,goal_mask,goal_image,episode_image):
        # goal concat token shape = (B,1,256), goal input token shape = (B,1,256)
        goal_mask_tensor = torch.as_tensor(goal_mask/255.0,dtype=torch.float32,device=self.device).permute(0,3,1,2).contiguous()
        goal_image_tensor = torch.as_tensor(goal_image/255.0,dtype=torch.float32,device=self.device).permute(0,3,1,2).contiguous()
        goal_token = self.goal_backbone(torch.concat((goal_image_tensor,goal_mask_tensor),dim=1)).unsqueeze(1)
        goal_concat_token = self.goal_concat_proj(goal_token)
        goal_input_token = self.goal_input_proj(goal_token)  

        # history image token shape = (B,64,512), and the episode input tokens are concated to (B,64,512+256)
        episode_image_tensor = torch.as_tensor(episode_image/255.0,dtype=torch.float32,device=self.device).permute(0,1,4,2,3).contiguous()
        B,T,C,H,W = episode_image_tensor.shape
        episode_image_tensor = episode_image_tensor.view(-1,C,H,W)
        epc_token = self.history_backbone(episode_image_tensor)
        epc_token = epc_token.view(B,T,epc_token.shape[-1])
        epc_token = torch.concat((epc_token,goal_concat_token.tile((1,epc_token.shape[1],1))),dim=-1)
        
        # add the position embedding
        pos_indice = torch.arange(self.max_token_length).expand(epc_token.shape[0],self.max_token_length).to(self.device)
        pos_embed = self.po_embedding(pos_indice)
        epc_token = epc_token + pos_embed
        tgt_mask = generate_square_subsequent_mask(self.max_token_length).to(self.device)
        out_token = self.dt_policy(tgt=epc_token,
                                   memory=goal_input_token,
                                   tgt_mask = tgt_mask)
        action_pred = self.action_head(out_token)
        distance_pred = self.distance_head(out_token)
        goal_pred = self.goal_head(out_token)
        return action_pred,distance_pred,goal_pred
