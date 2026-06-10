import os
os.environ['HF_HOME'] = '/home/tangjinqi/basketball/dino/models'   # 改成你的目录
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像,可选

import timm
# model = timm.create_model('convnext_tiny.dinov3_lvd1689m', pretrained=True).eval()
model = timm.create_model('convnext_small.dinov3_lvd1689m', pretrained=True).eval()