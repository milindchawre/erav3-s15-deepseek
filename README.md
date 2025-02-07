# DeepSeek Training Implementation

This repository contains a PyTorch implementation of the DeepSeek model architecture, achieved by enhancing the SmolLM2 base implementation with DeepSeek's architectural features. The model incorporates Multi-Head Latent Attention (MHLA) and Mixture of Experts (MoE) mechanisms from DeepSeek, and is trained on the "HuggingFaceTB/smollm-corpus" dataset (cosmopedia-v2 configuration) using streaming data loading.

## Model Architecture

### Key Components
- **Base Architecture**: Transformer-based decoder-only model
- **Model Size**: 774M parameters (1.48 GB)
- **Context Length**: 2048 tokens
- **Advanced Features**:
  - Multi-Head Latent Attention (MHLA) with 8x compression ratio
  - Mixture of Experts (MoE)
    - 8 expert networks
    - 1 shared expert
    - Top-2 expert routing
  - Rotary Position Embeddings (RoPE)
  - Group Query Attention (GQA)
  - RMSNorm for layer normalization

### Configuration
- Hidden size: 576
- Intermediate size: 1536
- Number of attention heads: 9
- Number of key-value heads: 3
- Number of layers: 30
- Vocabulary size: 49,152

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0.0
- transformers>=4.30.0
- datasets>=2.12.0
- numpy>=1.24.0
- accelerate>=0.20.0

## Project Structure
```
.
├── model.py # Model architecture implementation
├── train.py # Training script
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```

## Training Configuration
The model is trained with the following specifications:
- Training steps: 10000 + 100 extended steps
- Batch size: 8
- Sequence length: 2048
- Learning rate: 0.003 with linear warmup and decay
- Weight decay: 0.01
- Gradient clipping: 1.0
- Checkpointing interval: 1000 steps
- Validation/Generation interval: 500 steps
- bfloat16 precision (on supported devices)
- AdamW optimizer with β1=0.9, β2=0.95

## Features

- Multi-device support (CUDA, MPS, CPU)
- Streaming dataset loading for memory efficiency
- Automatic checkpoint resumption
- Regular model checkpointing
- Training progress metrics (loss, learning rate, step time, tokens/sec) displayed using print statements
- Generation samples during training
- FP32 gradient accumulation support
- Learning rate scheduling with warmup and decay
- Weight tying between input and output embeddings

## Training Process

During training, the model processes batches of data, computes the loss, and updates the model weights. The training script generates a log file that captures key metrics such as loss values, learning rates, and tokens processed per second. This log file is useful for monitoring the training progress and diagnosing any issues that may arise.

## Training Log File

The training log file is generated during the training process and contains detailed information about each training step. It includes metrics such as:
- Step number
- Loss value
- Learning rate
- Total step time
- Tokens processed per second

You can find the [training log file](./training.logs) in the project directory after running the training script.

## Potential Improvements

1. **Model Enhancements**
   - Implement flash attention for improved efficiency.
   - Add gradient checkpointing to reduce memory usage.
   - Support for model parallel training.

2. **Training Enhancements**
   - Incorporate a validation dataset for performance evaluation.
   - Implement early stopping based on validation loss.
   - Add a learning rate finder for optimal learning rate selection.
   - Support for distributed training across multiple devices.

3. **Code Organization**
   - Separate configuration settings into a dedicated config file.
   - Implement command-line interface (CLI) arguments for flexibility.
   - Improve error handling for robustness.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start training:
```bash
python train.py
```

The training script will:
- Automatically detect the best available device (CUDA, MPS, or CPU)
- Load the dataset in streaming mode
- Train for 10000 steps  # Updated
- Save checkpoints every 1000 steps with clear logging
- Generate text samples every 500 steps with clear logging
- Train for 100 additional steps  # Was incorrectly showing 50
- Save the final model

## Checkpoints and Model Saving

The training process saves several types of files:
- Regular checkpoints every 1000 steps: `checkpoints/step_*.pt`
- Checkpoint at step 10000: `checkpoints/step_10000.pt`  # Updated
- Final checkpoint at step 10100: `checkpoints/step_final_10100.pt`  # Updated
- Final model weights: `deepseek_final.pt`

Checkpoints contain:
- Model state
- Optimizer state
- Scheduler state
- Training step
- Loss value
- Configuration

## Training Metrics

During training, the following metrics are displayed for each step:
- Loss value
- Learning rate
- Step time in milliseconds
- Tokens per second
- Number of accumulated batches

Example output:
```
Step 1000 | Loss: 3.4567 | LR: 0.002345 | Total Step Time: 524.56ms | Tokens/sec: 31234.56 (accumulated over 8 batches)
```

## Resuming Training

The training automatically resumes from the latest checkpoint if interrupted. To resume training:
```bash
python train.py
```
The script will automatically detect and load the latest checkpoint.

## Model Output

The final model is saved in two formats:
1. Complete checkpoint with training state (`checkpoints/step_final_10100.pt`)
2. Model weights only (`deepseek_final.pt`)

## Training Progress Details

### Training Steps Breakdown
- Main training: 10000 steps  # Updated
- Extended training: 100 steps  # Updated
- Total training steps: 10100 steps  # Updated

### Per Step Processing
- Batch size per step: 8 sequences
- Sequence length: 2048 tokens
- Tokens processed per step: 8 × 2048 = 16,384 tokens
- Total tokens processed: 10100 steps × 16,384 tokens = 165,478,400 tokens  # Updated

### Dataset and Epochs
The training uses HuggingFace's streaming dataset approach:

```python
dataset = load_dataset(
"HuggingFaceTB/smollm-corpus",
"cosmopedia-v2",
streaming=True
)
```

- Dataset is loaded in streaming mode
- No explicit epoch boundaries
- Data streams continuously
- When reaching dataset end, streaming restarts automatically

### Checkpointing and Evaluation
- Checkpoints saved every 1000 steps with clear logging:
  ```
  ==================================================
  Saving checkpoint at step 1000
  ==================================================
  Checkpoint saved to: checkpoints/step_1000.pt
  ==================================================
  ```

- Text generation samples produced every 500 steps:
  ```
  ==================================================
  Generating text sample at step 500
  ==================================================
  Prompt: Once upon a time
  Generated text:
  [Generated text appears here]
  ==================================================
  ```

- Training metrics logged every step:
  ```
  Step 1000 | Loss: 3.4567 | LR: 0.002345 | Total Step Time: 524.56ms | Tokens/sec: 31234.56 (accumulated over 8 batches)
  ```

### Training Progress Tracking
1. **Step-based Progress**
   - Primary progress metric is steps completed
   - Two phases:
     * Main training progress (0-10000 steps)
     * Extended training progress (10000-10100 steps)

2. **Metrics Tracked Per Step**
   - Loss value
   - Learning rate
   - Step completion time
   - Tokens processed per second
   - Batch size

3. **Checkpoint Contents**
   - Model state
   - Optimizer state
   - Scheduler state
   - Current step
   - Current loss
   - Configuration

### Training Time Estimation
- Each step processes 16,384 tokens
- Processing speed varies by hardware:
  * GPU (CUDA): ~15,000-20,000 tokens/sec (updated for larger model)
  * Apple Silicon (MPS): ~5,000-10,000 tokens/sec
  * CPU: ~500-2,000 tokens/sec

### Memory Requirements
- Model size: ~774M parameters
- Peak memory usage varies by device:
  * Training on GPU: ~12-15GB VRAM (updated for larger model)
  * Training on CPU: ~20-25GB RAM
  * Gradient checkpointing can reduce memory usage if needed

### Output Files
During training, the following files are generated:
1. Regular checkpoints:
   - `checkpoints/step_1000.pt`
   - `checkpoints/step_2000.pt`
   - `checkpoints/step_3000.pt`
   - `checkpoints/step_4000.pt`
   - `checkpoints/step_5000.pt`
   - `checkpoints/step_6000.pt`
   - `checkpoints/step_7000.pt`
   - `checkpoints/step_8000.pt`
   - `checkpoints/step_9000.pt`
   - `checkpoints/step_10000.pt`
2. Final outputs:
   - `checkpoints/step_final_10100.pt` (complete training state)
   - `deepseek_final.pt` (final model weights only)

## License

[Specify your license here]

## Acknowledgments
- HuggingFace for the dataset and tokenizer
- DeepSeek for architectural inspirations
