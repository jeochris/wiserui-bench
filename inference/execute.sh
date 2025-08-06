#!/bin/bash

MODEL='gpt-4o' # gpt-4o, claude, o1, qwen2_5_vl_7b, qwen2_5_vl_32b, internvl_8b, internvl_38b, llava-next, llava-onevision
METHOD=('zero_shot' 'cocot' 'self_refine' 'ddcot' 'mad_each_debate' 'mad_moderate_extractive')  # Change this to the method you want to use
ROUND_FOR_MAD=(1 3)  # Change this to the number of rounds for MAD moderation
TASK=1 # 1 2

API_KEY='your_api_key'  # Replace with your actual API key
GPU_COUNT=1

for method in "${METHOD[@]}"; do
  if [[ "$method" == "mad_moderate_extractive" ]]; then
    for round in "${ROUND_FOR_MAD[@]}"; do
      python task.py $MODEL $method $round $TASK $API_KEY $GPU_COUNT
    done
  else
    python task.py $MODEL $method 1 $TASK $API_KEY $GPU_COUNT
  fi
done