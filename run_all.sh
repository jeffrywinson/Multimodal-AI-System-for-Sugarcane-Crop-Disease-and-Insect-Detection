#!/bin/bash

# ===================================================================================
# FINAL MASTER SCRIPT - UNIVERSAL (Local + Colab)
# This script automatically installs dependencies and runs the entire pipeline.
# It works on Windows (in Git Bash), macOS, Linux, AND Google Colab.
#
# It requires two arguments: path to disease image and path to insect image.
# ===================================================================================

# --- 0. Input Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_all.sh <path_to_disease_image> <path_to_insect_image>"
    exit 1
fi

DISEASE_IMAGE_PATH=$1
INSECT_IMAGE_PATH=$2


# --- 1. Universal Environment Detection ---
# This block makes the script work everywhere.
echo "--- Detecting Python Environment ---"
if [ -d "venv" ]; then
    # A local venv folder exists. Check for Windows vs Linux/macOS path.
    if [ -f "venv/Scripts/python" ]; then
        PYTHON_EXEC="./venv/Scripts/python" # Windows venv
        echo "Local Windows virtual environment detected."
    else
        PYTHON_EXEC="./venv/bin/python" # Linux/macOS venv
        echo "Local Linux/macOS virtual environment detected."
    fi
else
    # No venv folder found. Assume a Colab-like environment.
    # Use the globally available python3 command.
    PYTHON_EXEC="python3"
    echo "No local venv detected. Using global 'python3' (for Colab or similar environments)."
fi


# --- 2. Automatic Dependency Installation ---
echo ""
echo "--- Checking and Installing Dependencies from requirements.txt ---"
# The script now uses the universal $PYTHON_EXEC variable
$PYTHON_EXEC -m pip install -q -r requirements.txt
echo "--- Dependencies are up to date. ---"


# --- 3. Run the Main Application ---
echo ""
echo "--- Starting Multimodal Analysis ---"
echo "Disease Image: $DISEASE_IMAGE_PATH"
echo "Insect Image:  $INSECT_IMAGE_PATH"

# --- Define Farmer's Text Inputs (30 answers each) ---
DISEASE_SYMPTOMS="Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No,Yes,No"
INSECT_SYMPTOMS="Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes,No,No,Yes,Yes"

# --- Run All Four Models and Capture Outputs ---
echo "Running models..."
# All python calls now use the cross-platform $PYTHON_EXEC variable
YOLO_DISEASE_RESULT=$($PYTHON_EXEC detect_disease.py --image "$DISEASE_IMAGE_PATH")
YOLO_INSECT_RESULT=$($PYTHON_EXEC detect_insect.py --image "$INSECT_IMAGE_PATH")
TABNET_DISEASE_RESULT=$($PYTHON_EXEC tabnet/predict_disease_tabnet.py --answers "$DISEASE_SYMPTOMS")
TABNET_INSECT_RESULT=$($PYTHON_EXEC tabnet/predict_insect_tabnet.py --answers "$INSECT_SYMPTOMS")

# --- Run the Fusion Script ---
echo "--- Final Fused Output ---"
$PYTHON_EXEC fusion.py \
    --image_name "Combined Analysis" \
    --yolo_disease "$YOLO_DISEASE_RESULT" \
    --tabnet_disease "$TABNET_DISEASE_RESULT" \
    --yolo_insect "$YOLO_INSECT_RESULT" \
    --tabnet_insect "$TABNET_INSECT_RESULT"

echo "-------------------------------------"
echo "Analysis complete."