import os
import re
import time
import glob
import torch
import sys
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from datasets import load_dataset
from model import DeepSeek  # Update import
from torch.optim.lr_scheduler import LambdaLR

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    else:
        return torch.device("cpu"), "cpu"

def load_checkpoint(checkpoint_path):
    """Safely load checkpoint with error handling"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    except (RuntimeError, EOFError, Exception) as e:
        print(f"\nWarning: Failed to load checkpoint at {checkpoint_path}")
        print(f"Error: {str(e)}")
        print("Starting training from scratch...\n")
        return None

def get_lr_scheduler(optimizer, config):
    warmup_steps = config["optimizer"]["learning_rate_scheduler"]["lr_warmup_steps"]
    decay_start = config["optimizer"]["learning_rate_scheduler"]["lr_decay_starting_step"]
    decay_steps = config["optimizer"]["learning_rate_scheduler"]["lr_decay_steps"]
    base_lr = config["optimizer"]["learning_rate_scheduler"]["learning_rate"]
    min_lr = config["optimizer"]["learning_rate_scheduler"]["min_decay_lr"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        elif step < decay_start:
            return 1.0
        else:
            decay_ratio = (step - decay_start) / decay_steps
            decay_ratio = min(1.0, decay_ratio)
            return 1.0 - (1.0 - min_lr / base_lr) * decay_ratio

    return LambdaLR(optimizer, lr_lambda)

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory based on step number."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not checkpoints:
        return None
    
    # Extract step numbers and find the latest
    steps = [int(ckpt.split('step_')[-1].replace('.pt', '')) for ckpt in checkpoints]
    latest_checkpoint = checkpoints[steps.index(max(steps))]
    return latest_checkpoint

def save_final_model(model, save_path):
    """Save the final model in .pt format"""
    torch.save(model.state_dict(), save_path)
    print(f"Saved final model to: {save_path}")

def is_valid_loss(loss_value):
    """Check if loss value is valid"""
    return loss_value is not None and not torch.isnan(loss_value) and not torch.isinf(loss_value)

def main():
    # Get the best available device
    device, device_name = get_device()
    print(f"Using device: {device_name}")
    
    # Define configuration
    config = {
        "checkpoints": {
            "checkpoint_interval": 500,
            "checkpoints_path": "checkpoints",
            "checkpoints_path_is_shared_file_system": False,
            "resume_checkpoint_path": None,
            "save_final_state": False,
            "save_initial_state": False,
        },
        "general": {
            "benchmark_csv_path": None,
            "consumed_train_samples": None,
            "ignore_sanity_checks": True,
            "project": "deepseek",  # Update project name
            "run": "deepseek-774M",  # Update run name
            "seed": 8,
            "step": None,
        },
        "logging": {
            "iteration_step_info_interval": 1,
            "log_level": "info",
            "log_level_replica": "info",
        },
        "model": {
            "ddp_bucket_cap_mb": 25,
            "dtype": "bfloat16",
            "init_method": {
                "std": 0.041666666666666664,
            },
            "make_vocab_size_divisible_by": 1,
            "model_config": {
                "bos_token_id": 0,
                "eos_token_id": 0,
                "hidden_act": "silu",
                "hidden_size": 576,
                "initializer_range": 0.041666666666666664,
                "intermediate_size": 1536,
                "is_llama_config": True,
                "max_position_embeddings": 2048,
                "num_attention_heads": 9,
                "num_hidden_layers": 30,
                "num_key_value_heads": 3,
                "pad_token_id": None,
                "pretraining_tp": 1,
                "rms_norm_eps": 1.0e-05,
                "rope_interleaved": False,
                "rope_scaling": None,
                "rope_theta": 10000.0,
                "tie_word_embeddings": True,
                "use_cache": True,
                "vocab_size": 49152,
            },
        },
        "optimizer": {
            "accumulate_grad_in_fp32": True,
            "clip_grad": 1.0,
            "learning_rate_scheduler": {
                "learning_rate": 0.003,
                "lr_decay_starting_step": 1600000,
                "lr_decay_steps": 400000,
                "lr_decay_style": "linear",
                "lr_warmup_steps": 2000,
                "lr_warmup_style": "linear",
                "min_decay_lr": 0,
            },
            "optimizer_factory": {
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_eps": 1.0e-08,
                "name": "adamW",
                "torch_adam_is_fused": True,
            },
            "weight_decay": 0.01,
            "zero_stage": 0,
        },
        "parallelism": {
            "dp": 64,
            "expert_parallel_size": 1,
            "pp": 1,
            "pp_engine": "1f1b",
            "recompute_layer": False,
            "tp": 1,
            "tp_linear_async_communication": True,
            "tp_mode": "REDUCE_SCATTER",
            "tp_recompute_allgather": True,
        },
        "tokenizer": {
            "tokenizer_max_length": None,
            "tokenizer_name_or_path": "HuggingFaceTB/cosmo2-tokenizer",
            "tokenizer_revision": None,
        },
        "tokens": {
            "batch_accumulation_per_replica": 2,
            "limit_test_batches": 0,
            "limit_val_batches": 0,
            "micro_batch_size": 4,
            "sequence_length": 2048,
            "train_steps": 10000,
            "val_check_interval": 500,
        },
    }

    # Create checkpoints directory if it doesn't exist
    os.makedirs(config['checkpoints']['checkpoints_path'], exist_ok=True)

    # Load model configuration
    model_config = config["model"]["model_config"]
    
    # Set random seed
    torch.manual_seed(config["general"]["seed"])
    
    # Initialize model
    model = DeepSeek(model_config)  # Update model initialization
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: {total_params * 2 / (1024 * 1024):.2f} MB\n")
    
    # Initialize mixed precision training
    use_amp = device_name != "cpu"
    dtype = torch.float32
    if use_amp:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            scaler = None
            print("Using native bfloat16 mixed precision (no gradient scaling)")
        else:
            dtype = torch.float16
            scaler = GradScaler()
            print("Using float16 mixed precision with gradient scaling")
    else:
        scaler = None
        print("Using full precision (float32)")
    
    # Move model to device and set dtype
    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype)
    
    # Initialize tokenizer with padding token
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["tokenizer_name_or_path"])
    
    # Set padding token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set padding token to EOS token")
    
    # Update model config with pad token id
    model_config["pad_token_id"] = tokenizer.pad_token_id
    
    # Load dataset with streaming and specific config
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        streaming=True
    )
    train_dataset = dataset["train"]
    print("Loaded cosmopedia-v2 dataset in streaming mode")
    
    def tokenize_function(examples):
        # Process the batch of texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["tokens"]["sequence_length"],
            padding="max_length",
            return_tensors="pt"
        )
        
        # Stack the tensors properly
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": tokenized["attention_mask"].squeeze(0)  # Remove batch dimension
        }
    
    # Initialize step counter and total steps
    initial_total_steps = config["tokens"]["train_steps"]  # 10000
    extended_steps = 100  # Updated from 50
    final_total_steps = initial_total_steps + extended_steps
    step = 1  # Initialize step to 1
    
    # Try to find latest checkpoint
    checkpoint_files = glob.glob(os.path.join(config['checkpoints']['checkpoints_path'], 'step_*.pt'))
    latest_checkpoint = None
    
    if checkpoint_files:
        # Sort checkpoints by step number
        checkpoint_files.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
        latest_checkpoint = checkpoint_files[-1]
        
        print(f"\nFound checkpoint: {latest_checkpoint}")
        checkpoint = load_checkpoint(latest_checkpoint)
        
        if checkpoint is not None:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                step = checkpoint['step']  # Use loaded step directly
                print(f"Resumed from step {step}")
            except Exception as e:
                print(f"Warning: Failed to restore checkpoint state: {str(e)}")
                print("Starting training from scratch at step 1...")
                step = 1  # Reset to 1 if checkpoint loading fails
    
    # Initialize dataset iterator
    train_iter = iter(train_dataset.map(
        tokenize_function,
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=config["tokens"]["micro_batch_size"]
    ))
    
    # Skip batches if resuming from checkpoint
    if step > 1:
        print(f"Skipping {step * config['tokens']['batch_accumulation_per_replica']} batches to resume position...")
        for _ in range(step * config["tokens"]["batch_accumulation_per_replica"]):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataset.map(
                    tokenize_function,
                    remove_columns=train_dataset.column_names,
                    batched=True,
                    batch_size=config["tokens"]["micro_batch_size"]
                ))
                next(train_iter)
        print("Done skipping batches")
    
    # Initialize optimizer
    use_fused = config["optimizer"]["optimizer_factory"]["torch_adam_is_fused"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate_scheduler"]["learning_rate"],
        betas=(
            config["optimizer"]["optimizer_factory"]["adam_beta1"],
            config["optimizer"]["optimizer_factory"]["adam_beta2"]
        ),
        eps=config["optimizer"]["optimizer_factory"]["adam_eps"],
        weight_decay=config["optimizer"]["weight_decay"],
        fused=use_fused
    )
    
    # Initialize scheduler
    scheduler = get_lr_scheduler(optimizer, config)
    
    # Training parameters
    batch_size = config["tokens"]["micro_batch_size"]
    accum_steps = config["tokens"]["batch_accumulation_per_replica"]
    save_steps = 1000
    eval_steps = 500
    grad_clip = 1.0
    
    print("\nStarting training...")
    print(f"Total steps: {final_total_steps}")
    print(f"Device: {device_name}")
    print(f"{'='*50}\n")
    
    while step <= final_total_steps:  # Changed < to <= to include final step
        step_start_time = time.time()
        accumulated_loss = 0
        
        # Training step
        for accum_step in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataset.map(
                    tokenize_function,
                    remove_columns=train_dataset.column_names,
                    batched=True,
                    batch_size=config["tokens"]["micro_batch_size"]
                ))
                batch = next(train_iter)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
            
            if use_amp:
                with autocast(dtype=dtype):
                    outputs = model(input_ids, attention_mask)
                    loss = torch.nn.functional.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        input_ids.view(-1)
                    )
                    loss = loss / accum_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(input_ids, attention_mask)
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    input_ids.view(-1)
                )
                loss = loss / accum_steps
                loss.backward()
            
            accumulated_loss += loss.item() * accum_steps
        
        # Optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        scheduler.step()
        optimizer.zero_grad()
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Calculate step time and tokens/sec
        step_time = time.time() - step_start_time
        step_time_ms = step_time * 1000
        tokens_per_second = (config["tokens"]["micro_batch_size"] * config["tokens"]["sequence_length"]) / step_time
        
        # Print step info
        print(f"Step {step}/{final_total_steps} | Loss: {accumulated_loss:.4f} | LR: {current_lr:.6f} | "
                    f"Total Step Time: {step_time_ms:.2f}ms | "
                    f"Tokens/sec: {tokens_per_second:.2f} (accumulated over {config['tokens']['micro_batch_size']} batches)")
        
        # Text generation at step 500, 1000, 1500, etc.
        if (step % eval_steps) == 0:
            print(f"\n{'='*50}")
            print(f"Generating text sample at step {step}")
            print(f"{'='*50}")
            
            model.eval()
            with torch.no_grad():
                prompt = "Once upon a time"
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                generated = model.generate(
                    input_ids,
                    max_length=200,
                    min_length=50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                print(f"Prompt: {prompt}")
                print(f"Generated text:\n{generated_text}")
                print(f"{'='*50}\n")
            model.train()
        
        # Checkpointing at step 1000, 2000, 3000, etc.
        if (step % save_steps) == 0:
            print(f"\n{'='*50}")
            print(f"Saving checkpoint at step {step}")
            print(f"{'='*50}")
            
            checkpoint_path = os.path.join(config['checkpoints']['checkpoints_path'], f"step_{step}.pt")
            final_checkpoint_path = os.path.join(config['checkpoints']['checkpoints_path'], f"step_final_{step}.pt")
            final_model_path = "deepseek_final.pt"  # Updated from smollm2_final.pt
            
            try:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': accumulated_loss,
                    'config': config,
                }, checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")
                print(f"{'='*50}\n")
                
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {str(e)}")
                print(f"{'='*50}\n")
        
        # Increment step counter at the end
        step += 1
    
    print("\nTraining completed!")
    
    # Save final checkpoint and model
    final_checkpoint_path = f"{config['checkpoints']['checkpoints_path']}/step_final_{final_total_steps}.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': accumulated_loss,
        'config': config,
    }, final_checkpoint_path)
    print(f"Saved final checkpoint: {final_checkpoint_path}")
    
    # Save final model in .pt format
    model_save_path = "deepseek_final.pt"  # Update save path
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved final model to: {model_save_path}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()