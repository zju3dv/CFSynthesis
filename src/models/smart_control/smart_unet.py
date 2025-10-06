from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.utils import is_torch_version
import matplotlib.pyplot as plt

def upblock3d_forward(self):
    def forward(
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        encoder_hidden_states=None,
        c_predictor =None
    ) -> torch.FloatTensor:
        count =0 
        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            c_half = res_hidden_states.shape[1]//2
            h_1 = torch.cat([hidden_states, res_hidden_states[:,:c_half] + res_hidden_states[:,c_half:]], dim=1)
            h_4 = torch.cat([hidden_states, res_hidden_states[:,:c_half]], dim=1)
            c = torch.sigmoid(c_predictor[count](torch.cat([h_1 , h_4, res_hidden_states[:,c_half:]], dim=1)))
            res_hidden_states  = res_hidden_states[:,:c_half] + c* res_hidden_states[:,c_half:]
            count = count+1

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
                
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                    )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = (
                    motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )
                    if motion_module is not None
                    else hidden_states
                )


        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
    return forward



def crossattnupblock3d_forward(self):
    def forward(
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        c_predictor =None

    ) -> torch.FloatTensor:
    
        count =0
        for i, (resnet, attn, motion_module) in enumerate(
            zip(self.resnets, self.attentions, self.motion_modules)
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            
            c_half = res_hidden_states.shape[1]//2
            h_1 = torch.cat([hidden_states, res_hidden_states[:,:c_half] + res_hidden_states[:,c_half:]], dim=1)
            h_4 = torch.cat([hidden_states, res_hidden_states[:,:c_half]], dim=1)
            c = torch.sigmoid(c_predictor[count](torch.cat([h_1 , h_4, res_hidden_states[:,c_half:]], dim=1)))
            res_hidden_states  = res_hidden_states[:,:c_half] + c* res_hidden_states[:,c_half:]
            count = count+1

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states.requires_grad_(),
                        temb,
                        encoder_hidden_states,
                    )
                    
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # add motion module
                hidden_states = (
                    motion_module(
                        hidden_states, temb, encoder_hidden_states=encoder_hidden_states
                    )
                    if motion_module is not None
                    else hidden_states
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
    return forward