#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:11:01 2024

@author: xmw5190
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:59:46 2024

@author: xmw5190
"""

import open_clip
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
import timm


clip_encoder = torch.load("/data/xiaochen/CSP_checkpoints/csp_clip.pt").cuda()
print(type(clip_encoder.visual.trunk))
image_forward_outs  = clip_encoder.visual.trunk.forward_features(torch.rand(8,3,224,224).cuda())
# print(type(clip_encoder.visual))
# clip_encoder.visual.output_tokens = True
# print(clip_encoder.visual.output_tokens)
# # .visual_encoder = clip_encoder
# # print (dir(clip_encoder))
# a = clip_encoder.visual(torch.rand(3,3,224,224).cuda())
print(image_forward_outs.shape)
# print(type(clip_encoder))