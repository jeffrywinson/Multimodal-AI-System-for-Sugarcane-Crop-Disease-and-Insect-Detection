#!/bin/bash

echo "--- Starting Multimodal Analysis (Final Version) ---"

# 1. DEFINE INPUTS
IMAGE_FILE="sugarcane_test_01.jpg"

# CORRECTED: Provide exactly 30 comma-separated answers for the disease questions
DISEASE_SYMPTOMS="Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No"

# CORRECTED: Provide exactly 30 comma-separated answers for the insect questions
INSECT_SYMPTOMS="Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes"


# 2. RUN MODELS AND CAPTURE OUTPUTS
echo "Running models..."

# Use the dummy scripts for now. Replace with real scripts later.
YOLO_DISEASE_RESULT=$(python dummy_detect_disease.py --img "$IMAGE_FILE")
YOLO_INSECT_RESULT=$(python dummy_detect_insect.py --img "$IMAGE_FILE")

# Run your actual TabNet prediction scripts
# The python command needs the ./venv/Scripts/ prefix to work reliably in your setup
TABNET_DISEASE_RESULT=$(./venv/Scripts/python predict_disease_tabnet.py --answers "$DISEASE_SYMPTOMS")
TABNET_INSECT_RESULT=$(./venv/Scripts/python predict_insect_tabnet.py --answers "$INSECT_SYMPTOMS")


# 3. RUN THE FUSION SCRIPT
echo "--- Final Fused Output ---"
# The python command needs the ./venv/Scripts/ prefix here as well
./venv/Scripts/python fusion.py \
    --image_name "$IMAGE_FILE" \
    --yolo_disease "$YOLO_DISEASE_RESULT" \
    --tabnet_disease "$TABNET_DISEASE_RESULT" \
    --yolo_insect "$YOLO_INSECT_RESULT" \
    --tabnet_insect "$TABNET_INSECT_RESULT"

echo "-------------------------------------"
echo "Analysis complete."