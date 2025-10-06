import torch.nn as nn
import torch
from .smart_unet import upblock3d_forward,crossattnupblock3d_forward
from .resnet import InflatedConv3d

def init_smartcontrol(unet,smart_ckpt=None):
    c_13_list = [1280,2560,2560,2560,2560,2560,2560,2560,1280,1280,1280,640,640]
    c_skip_list = [1280,2560,2560,2560,2560,2560,1280,1280,1280,640,640,640,640]
    c_pre_list = nn.ModuleList()
    count = 0
    for c in c_13_list:
        c_skip = c_skip_list[count]
        layers=[]
        if count==0:
            c_init = c+c_skip+c_skip
        else:
            c_init = c+c_skip+c_skip//2
            
        c_block = nn.Sequential(
                    InflatedConv3d(c_init, c_init//4, 3, padding=1),
                    nn.SiLU(),
                    InflatedConv3d(c_init//4, c_init//8, 3, padding=1),
                    nn.SiLU(),
                    InflatedConv3d(c_init//8, 1, 3, padding=1))
                    
        count = count+1

        c_pre_list.append(c_block)

    state_dict={}
    # pretrained_weights = torch.load(smart_ckpt, map_location="cpu")

    # for k, v in pretrained_weights.items():
    #     name = k[11:] 
    #     state_dict[name] = v 
    unet.c_pre_list = c_pre_list.to(dtype=torch.float32)
    # unet.c_pre_list.load_state_dict(state_dict)

def register_module(net):
    for name, subnet in net.named_children():
        if subnet.__class__.__name__ == 'CrossAttnUpBlock3D': 
            subnet.forward = crossattnupblock3d_forward(subnet)
        if subnet.__class__.__name__ == 'UpBlock3D': 
            subnet.forward = upblock3d_forward(subnet)
        elif hasattr(subnet, 'children'):
            register_module(subnet)


def register_unet(unet,smart_ckpt=None):
    init_smartcontrol(unet,smart_ckpt)
    register_module(unet)
    return unet