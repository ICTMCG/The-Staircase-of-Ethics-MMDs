#!/bin/bash
# Run the multi-layer dilemma generator
# Usage: bash multilayer_dilemma.sh

set -e  # exit on error

# Set dataset path
INPUT_FILE="./data/moral_story_dataset.json",
OUTPUT_FILE="./data/multilayer_dilemma.json",

# Create output dir if not exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run Python script
python multilayer_dilemma.py \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --batch_size 5 \
  --max_workers 5 \
  --process_range 100 \
  --model gpt-4o
