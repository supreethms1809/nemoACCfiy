# FSDP (Fully Sharded Data Parallel) Support

This document explains how to use FSDP for multi-node multi-GPU training with the ModularModel.

## Overview

FSDP (Fully Sharded Data Parallel) is a PyTorch distributed training strategy that shards model parameters, gradients, and optimizer states across multiple GPUs and nodes. This allows training of very large models that don't fit on a single GPU.

## Configuration

### 1. Basic FSDP Configuration

Add the following to your `config.yaml` under each training stage:

```yaml
training_stages:
  stage1:
    # ... other config ...
    
    # Distributed training configuration
    distributed:
      strategy: "fsdp"  # or "auto" with fsdp.enabled: true
      num_nodes: 1
      devices: 8  # Number of GPUs per node
      sync_batchnorm: false
      
      # FSDP specific configuration
      fsdp:
        enabled: true
        cpu_offload: false
        sharding_strategy: "FULL_SHARD"  # Options: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
        backward_prefetch: "BACKWARD_PRE"  # Options: "BACKWARD_PRE", "BACKWARD_POST", "NONE"
        forward_prefetch: false
        limit_all_gathers: true
        activation_checkpointing: true
        use_orig_params: false
```

### 2. Pre-configured Examples

The config.yaml includes several pre-configured FSDP setups:

#### Single Node Multi-GPU
```yaml
distributed_examples:
  single_node_fsdp:
    strategy: "fsdp"
    num_nodes: 1
    devices: 8  # 8 GPUs on single node
    fsdp:
      enabled: true
      cpu_offload: false
      sharding_strategy: "FULL_SHARD"
      activation_checkpointing: true
```

#### Multi-Node Multi-GPU
```yaml
distributed_examples:
  multi_node_fsdp:
    strategy: "fsdp"
    num_nodes: 4  # 4 nodes
    devices: 8    # 8 GPUs per node
    fsdp:
      enabled: true
      cpu_offload: true  # Enable CPU offload for very large models
      sharding_strategy: "FULL_SHARD"
      activation_checkpointing: true
```

#### Large Scale Training
```yaml
distributed_examples:
  large_scale_fsdp:
    strategy: "fsdp"
    num_nodes: 8
    devices: 8
    fsdp:
      enabled: true
      cpu_offload: true
      sharding_strategy: "FULL_SHARD"
      backward_prefetch: "BACKWARD_PRE"
      forward_prefetch: true
      limit_all_gathers: true
      activation_checkpointing: true
      use_orig_params: false
```

## Usage

### 1. Using the Example Script

```bash
# Single node, 8 GPUs
python fsdp_example.py --fsdp-config single_node_fsdp --devices 8

# Multi-node, 4 nodes with 8 GPUs each
python fsdp_example.py --fsdp-config multi_node_fsdp --num-nodes 4 --devices 8

# Large scale training
python fsdp_example.py --fsdp-config large_scale_fsdp --num-nodes 8 --devices 8
```

### 2. Manual Configuration

1. Update your `config.yaml` with the desired FSDP configuration
2. Run training using the production mode:

```python
from src.nemo.ModularModelstage1_NTPtraining import train_production_mode

trainer, module = train_production_mode(
    model_config_key="model_config_1.7B",
    stage="stage1",
    devices=8,  # Will be overridden by config if specified
    precision="bf16-mixed"
)
```

### 3. Command Line Training

```bash
# Using the main training script with FSDP
python -m src.nemo.ModularModelstage1_NTPtraining \
    --mode production \
    --stage stage1 \
    --model-config model_config_1.7B \
    --devices 8 \
    --precision bf16-mixed
```

## FSDP Configuration Options

### Strategy Options
- `"auto"`: PyTorch Lightning automatically chooses the best strategy
- `"fsdp"`: Use FSDP strategy
- `"ddp"`: Use DDP strategy (alternative to FSDP)

### Sharding Strategy
- `"FULL_SHARD"`: Shard parameters, gradients, and optimizer states (most memory efficient)
- `"SHARD_GRAD_OP"`: Shard gradients and optimizer states only
- `"NO_SHARD"`: Don't shard (equivalent to DDP)

### CPU Offload
- `cpu_offload: true`: Offload parameters to CPU when not in use (saves GPU memory)
- `cpu_offload: false`: Keep all parameters on GPU (faster but uses more memory)

### Prefetching
- `backward_prefetch`: When to prefetch parameters for backward pass
  - `"BACKWARD_PRE"`: Prefetch before backward pass (recommended)
  - `"BACKWARD_POST"`: Prefetch after backward pass
  - `"NONE"`: No prefetching
- `forward_prefetch`: Whether to prefetch for forward pass

### Other Options
- `limit_all_gathers: true`: Limit concurrent all-gather operations (recommended)
- `activation_checkpointing: true`: Use gradient checkpointing to save memory
- `use_orig_params: false`: Use original parameter format (recommended for most cases)

## Multi-Node Setup

For multi-node training, you need to:

1. **Set up the cluster**: Ensure all nodes can communicate
2. **Configure networking**: Set up proper network interfaces
3. **Set environment variables**:
   ```bash
   export MASTER_ADDR=<master_node_ip>
   export MASTER_PORT=<port>
   export WORLD_SIZE=<total_gpus>
   export RANK=<node_rank>
   export LOCAL_RANK=<local_gpu_rank>
   ```

4. **Launch training on each node**:
   ```bash
   # On node 0
   python fsdp_example.py --fsdp-config multi_node_fsdp --num-nodes 4 --devices 8
   
   # On node 1, 2, 3 (with appropriate RANK and LOCAL_RANK)
   python fsdp_example.py --fsdp-config multi_node_fsdp --num-nodes 4 --devices 8
   ```

## Memory Optimization Tips

1. **Use CPU offload** for very large models:
   ```yaml
   fsdp:
     cpu_offload: true
   ```

2. **Enable activation checkpointing**:
   ```yaml
   fsdp:
     activation_checkpointing: true
   ```

3. **Use gradient accumulation** to increase effective batch size:
   ```yaml
   training:
     gradient_accumulation_steps: 4
   ```

4. **Adjust batch size** based on available memory:
   ```yaml
   training:
     batch_size: 2  # Smaller batch size for large models
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable CPU offload
2. **Slow Training**: Check network bandwidth between nodes
3. **Hanging**: Ensure proper environment variables are set
4. **Import Errors**: Make sure PyTorch Lightning and FSDP are properly installed

### Debug Mode

Enable debug logging to troubleshoot issues:

```yaml
logging:
  debug: true
  log_file: "training.log"
```

### Performance Monitoring

Monitor training performance with:

```yaml
wandb:
  use_wandb: true
  project: "accfiy-fsdp"
  tags: ["fsdp", "multi-node"]
```

## Example Commands

```bash
# Single node, 4 GPUs
python fsdp_example.py --fsdp-config single_node_fsdp --devices 4

# Multi-node, 2 nodes with 8 GPUs each
python fsdp_example.py --fsdp-config multi_node_fsdp --num-nodes 2 --devices 8

# Large model with CPU offload
python fsdp_example.py --fsdp-config large_scale_fsdp --num-nodes 4 --devices 8

# Stage 2 training with FSDP
python fsdp_example.py --stage stage2 --fsdp-config multi_node_fsdp --devices 8
```
