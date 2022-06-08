#! /bin/bash

root_dir=$PWD
CONFIG_FILE="${root_dir}/configs/lgpma_pub.py"
CHECKPOINT_FILE="${root_dir}/models/maskrcnn-lgpma-pub-e12-pub.pth"
OUTPUT_FILE="lgpma.onnx"
INPUT_IMAGE_PATH="images/1.png"
# IMAGE_SHAPE="1376 1248"
IMAGE_SHAPE="688 624"
OPSET_VERSION=11

python pytorch2onnx.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --input-img ${INPUT_IMAGE_PATH} \
    --shape ${IMAGE_SHAPE} \
    --opset-version ${OPSET_VERSION} \
    --dynamic-export \
    --verify \
    --simplify