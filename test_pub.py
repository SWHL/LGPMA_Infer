"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_pub.py
# Abstract       :    Script for inference

# Current Version:    1.0.1
# Date           :    2021-09-23
##################################################################################################
"""
from pathlib import Path

import cv2
from davarocr.davar_common.apis import inference_model, init_model


root_dir = Path(__file__).resolve().parent

vis_dir = str(root_dir / 'output')

# path setting
config_file = str(root_dir / 'configs/lgpma_pub.py')
checkpoint_file = str(root_dir / 'models/maskrcnn-lgpma-pub-e12-pub.pth')

model = init_model(config_file, checkpoint_file, 'cpu')

# 可视化单张图像的cell 框结果
img_path = 'images/table.jpg'
img = cv2.imread(img_path)

result = inference_model(model, img)[0]

img_name = img_path.split("/")[-1]
bboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]]
            for b in result['content_ann']['bboxes'] if len(b) > 0]
for box in bboxes:
    for j in range(0, len(box), 2):
        cv2.line(img, (box[j], box[j + 1]),
                    (box[(j + 2) % len(box)],
                    box[(j + 3) % len(box)]),
                    (0, 0, 255), 1)

cv2.imwrite(str(Path(vis_dir) / img_name), img)
print('ok')