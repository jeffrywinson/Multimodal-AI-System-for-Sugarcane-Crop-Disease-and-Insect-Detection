#!/bin/bash

# This script runs the final, integrated multimodal pipeline.
# It requires two arguments: the path to the disease image and the path to the insect image.

if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_all.sh <path_to_disease_image> <path_to_insect_image>"
    exit 1
fi

DISEASE_IMAGE_PATH=$1
INSECT_IMAGE_PATH=$2

echo "--- Starting Final Multimodal Analysis ---"
echo "Disease Image: $DISEASE_IMAGE_PATH"
echo "Insect Image:  $INSECT_IMAGE_PATH"

# --- Define Farmer's Text Inputs (30 answers each) ---
DISEASE_SYMPTOMS="Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No"
INSECT_SYMPTOMS="Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes"

# --- Run All Four Models and Capture Outputs ---
echo "Running models..."

# Use the real prediction scripts
# The python command needs the ./venv/Scripts/ prefix to work reliably in your setup
YOLO_DISEASE_RESULT=$(./venv/Scripts/python detect_disease.py --image "$DISEASE_IMAGE_PATH")
YOLO_INSECT_RESULT=$(./venv/Scripts/python detect_insect.py --image "$INSECT_IMAGE_PATH")

# Point to the TabNet scripts inside the new 'tabnet' folder
TABNET_DISEASE_RESULT=$(./venv/Scripts/python tabnet/predict_disease_tabnet.py --answers "$DISEASE_SYMPTOMS")
TABNET_INSECT_RESULT=$(./venv/Scripts/python tabnet/predict_insect_tabnet.py --answers "$INSECT_SYMPTOMS")

# --- Run the Fusion Script ---
echo "--- Final Fused Output ---"
# The fusion script is in the main directory
./venv/Scripts/python fusion.py \
    --image_name "Combined Analysis" \
    --yolo_disease "$YOLO_DISEASE_RESULT" \
    --tabnet_disease "$TABNET_DISEASE_RESULT" \
    --yolo_insect "$YOLO_INSECT_RESULT" \
    --tabnet_insect "$TABNET_INSECT_RESULT"

echo "-------------------------------------"
echo "Analysis complete."