#!/bin/bash
# Get arguments
CONFIG=$1
MODE=${2:-train}
NPROC_PER_NODE=${3:-1}

# Default values
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching distributed training..."
echo "Config: $CONFIG"
echo "Mode: $MODE"
echo "GPUs per node: $NPROC_PER_NODE"

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    --config $CONFIG \
    --mode $MODE
