#!/bin/bash

WORKLOADS_DIR="/home/zhanghf/projects/cimloop/workspace/models/workloads"
SCRIPT_PATH="/home/zhanghf/projects/cimloop/workspace/scripts/run_test.py"

# Define your specific workloads here
WORKLOADS=(
    "resnet18"
    "vgg16"
    # "alexnet"
    # "mobilenet_v3"
    # "resnet18_condensed"
    # "vgg16_condensed"
    # Add more workloads as needed
)

# Create results directory if it doesn't exist
mkdir -p test_results

# Loop through defined workloads
for workload in "${WORKLOADS[@]}"; do
    workload_path="$WORKLOADS_DIR/$workload"
    
    # Check if workload directory exists
    if [ ! -d "$workload_path" ]; then
        echo "Warning: Workload directory $workload_path not found - skipping"
        continue
    fi
    
    echo "Processing workload: $workload"
    
    echo "Running holylight architecture tests for $workload..."
    python "$SCRIPT_PATH" --arch holylight_date_2019 --network "$workload" > "workspace/logs/${workload}_holylight.log" 2>&1

    echo "Running proposed architecture tests for $workload..."
    python "$SCRIPT_PATH" --arch proposed_mrr --network "$workload" > "workspace/logs/${workload}_mrr.log" 2>&1
    
    echo "Running proposed 1bit architecture tests for $workload..."
    python "$SCRIPT_PATH" --arch proposed_mrr_1bit_input --network "$workload" > "workspace/logs/${workload}_mrr_1bit_input.log" 2>&1

    echo "Running proposed delay line architecture tests for $workload..."
    python "$SCRIPT_PATH" --arch proposed_mrr_1bit_input_delay_line --network "$workload" > "workspace/logs/${workload}_mrr_1bit_input_delay_line.log" 2>&1

done

echo "All specified tests completed. Results logged to test_results directory."
