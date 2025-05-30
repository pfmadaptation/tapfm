import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import utils_tapfm
import torch.nn.functional as F
import json
import numpy as np
import math

def setup_encoder(args):
    """
    Setup the encoder (tile) model, its optimizer, and gradient scaler.
    
    Args:
        args: Parsed command line arguments containing model configuration
        
    Returns:
        tuple: (model, optimizer, gradient scaler) for encoder training
    """
    # Initialize the encoder model
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        tile_model = utils_tapfm.get_pfm(args.pfm)
        # get head info for the encoder (PFM) model
        num_heads, head_dim, hidden_dim = utils_tapfm.get_head_info(tile_model)

        tile_model.ndim = hidden_dim  # Set the feature dimension
        args.ndim = hidden_dim  # Store for aggregator setup
        tile_model = tile_model.to(args.gpu)
    # tile_model.gradient_checkpointing = True
    # Setup optimizer with parameter groups
    params_groups = utils_tapfm.get_params_groups(tile_model)
    if args.optimizer == 'sgd':
        tile_optimizer = optim.SGD(
            params_groups, 
            lr=0., 
            momentum=args.momentum,
            dampening=0,
            nesterov=True
        )
    elif args.optimizer == 'adam':
        tile_optimizer = optim.Adam(params_groups)
    elif args.optimizer == 'adamw':
        tile_optimizer = optim.AdamW(params_groups, lr=args.lr)
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    print(f"Initialized encoder model with {sum(p.numel() for p in tile_model.parameters())} parameters")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        tile_optimizer,
        T_0=args.nepochs,           # First cycle length = 10 epochs
        T_mult=1,         # Each cycle is 2x longer than the previous
        eta_min=1e-8      # Minimum learning rate
        )

    # Resume from checkpoint if available
    start_epoch = utils_tapfm.restart_from_checkpoint(
        os.path.join(args.outdir, 'checkpoint_tile.pth'),
        tile_model=tile_model,
        tile_optimizer=tile_optimizer,
        scaler=scaler
    )

    return tile_model, tile_optimizer, scaler, scheduler, start_epoch

def setup_aggregator(args):
    """
    Setup the aggregator (slide) model, its optimizer, and loss criterion.
    
    Args:
        args: Parsed command line arguments containing model configuration
        
    Returns:
        tuple: (model, optimizer, criterion) for aggregator training
    """
    # Initialize the aggregator model
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        slide_model = utils_tapfm.tapfm_aggregator(ndim=args.ndim, n_classes=len(args.target_list), dropout_rate=0.50) 
        slide_model = slide_model.to(args.gpu)
    
    # Setup loss criterion with class weights if specified
    # compute class weigths and alpha values for multi-label loss function
    # alpha_values = (1 - proportion of positive slides) in each class
    alpha_values = [0.85, 0.82] # (1 - proportion of positive slides) in each class
    alpha_values = torch.tensor(alpha_values, dtype=torch.float)

    cw = [6.72, 5.52] #BLCA -- PIK3CA and FGFR3
    
    # cw = [1, 1]
    cw = torch.tensor(cw, dtype=torch.float)
    cw = cw * (len(args.target_list)/ cw.sum())

    # Initialize with imbalance-optimized parameters for your class distribution
    criterion = utils_tapfm.WeightedCrossEntropyMILLoss(class_weights = cw, alpha = alpha_values)

    args.pos_proportion = [1-alpha for alpha in alpha_values]
    # Setup optimizer with parameter groups
    params_groups = utils_tapfm.get_params_groups(slide_model)
    if args.optimizer == 'sgd':
        slide_optimizer = optim.SGD(
            params_groups,
            lr=0.,
            momentum=args.momentum,
            dampening=0,
            nesterov=True
        )
    elif args.optimizer == 'adam':
        slide_optimizer = optim.Adam(params_groups)
    elif args.optimizer == 'adamw':
        slide_optimizer = optim.AdamW(params_groups, lr=args.lr*10)#, lr=args.lr * 100.)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            slide_optimizer,
            T_0=args.nepochs,           # First cycle length = 2 epochs
            T_mult=2,         # Each cycle is 2x longer than the previous
            eta_min=1e-7      # Minimum learning rate
            )     
    

    # Resume from checkpoint if available
    start_epoch = utils_tapfm.restart_from_checkpoint(
        os.path.join(args.outdir, 'checkpoint_slide.pth'),
        slide_model=slide_model,
        slide_optimizer=slide_optimizer
    )
    print(f"Initialized aggregator model with {sum(p.numel() for p in slide_model.parameters())} parameters")
    
    return slide_model, slide_optimizer, criterion, scheduler, start_epoch


def encoder_forward(tile_model, inputs, args):
    """Compute features using encoder"""
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        inputs = inputs.to(args.gpu)
        features = tile_model.forward(inputs)
        features = features.contiguous()
    # Add debug print
        # print(f"Feature stats before float conversion - Min: {features.min():.6f}, Max: {features.max():.6f}")
    features = features.float()
    # Add debug print
    # print(f"Feature stats after float conversion - Min: {features.min():.6f}, Max: {features.max():.6f}")

    # Add normalization if needed
    # nfeatures = F.normalize(features, p=2, dim=1)  # L2 normalization
    # print(f"Feature stats after normalization - Min: {nfeatures.min():.6f}, Max: {nfeatures.max():.6f}")
    return features

def aggregator_step(slide_model, slide_optimizer, criterion, storage, dataset, batch_idx, size_dset, args):
    # batch_start = time.time()
    """Run aggregator forward and backward pass"""
    # Get features from storage
    features = storage.get_features()
    attention_weights = storage.get_attention()

    # Create new tensor with proper gradient tracking
    features = features.clone().detach().requires_grad_(True)
    attention_weights = attention_weights.clone().detach().requires_grad_(True)
    # features.requires_grad_()
    # Debug prints
    # print(f"Features stats - Min: {features.min():.4f}, Max: {features.max():.4f}")
    # print(f"Attention stats - Min: {attention_weights.min():.4f}, Max: {attention_weights.max():.4f}")

    
    # Forward pass
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        _, output = slide_model(features, attention_weights)
        torch.cuda.synchronize()  # Synchronize after forward pass
        # print(f"Model output: {output.softmax(dim=-1)}")  # Check probabilities
        # label = torch.LongTensor([dataset.get_target(batch_idx)]).to(args.gpu)

        prediction = torch.sigmoid(output)
        label = dataset.get_target(batch_idx).to(args.gpu)
        # loss, prediction = criterion(output, label, batch_idx, size_dset)

        loss, class_loss  = criterion(output, label)
    # loss, prediction = criterion(output, label, attention_weights, batch_idx, size_dset)
    torch.cuda.synchronize()  # Synchronize after forward pass

    # Backward pass
    slide_optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()  
   

    grad_norm = features.grad.norm().item()
    attention_grad_norm = attention_weights.grad.norm().item()
    # print(f"Feature Gradient norm: {grad_norm:.4f}")
    # print(f"Feature Gradient - Min: {features.grad.min().item():.6f}, Mean: {features.grad.mean().item():.6f}, Max: {features.grad.max().item():.6f}")
    
    # print(f"Attention  Gradient Norm: {attention_grad_norm:.4f}")
    # print(f"Attention Gradient - Min: {attention_weights.grad.min().item():.6f}, Mean: {attention_weights.grad.mean().item():.6f}, Max: {attention_weights.grad.max().item():.6f}")
    # # print(f"Gradient values: {features.grad}")

    # Store gradients for encoder
    storage.store_gradients(features.grad.detach())
    storage.store_attention_grads(attention_weights.grad.detach())
    torch.cuda.synchronize()  # Synchronize after forward pass
    # Update aggregator
    slide_optimizer.step()
    torch.cuda.synchronize()  # Synchronize after forward pass
    # batch_time = time.time() - batch_start
    # print(f"Batch {batch_idx+1}/{size_dset} - Loss: {loss:.4f} - Time: {batch_time:.2f}s")
    return loss.item(), prediction, label

def encoder_backward(tile_model, tile_optimizer, scaler, features, attention_weights, storage, args):
    """Run encoder backward pass"""
    # Get gradients from storage
    gradients = storage.get_gradients()

    attention_grads = storage.get_attention_grads()

    # Print statistics to debug
    # print(f"Feature stats - Min: {features.min():.6f}, Max: {features.max():.6f}")
    # print(f"Feature grads stats - Min: {gradients.min():.6f}, Max: {gradients.max():.6f}")
    # print(f"Attention weights stats - Min: {attention_weights.min():.6f}, Max: {attention_weights.max():.6f}")
    # print(f"Attention grads stats - Min: {attention_grads.min():.6f}, Max: {attention_grads.max():.6f}")
    # print(f"Element-wise product stats - Min: {(attention_weights * attention_grads).min():.6f}, Max: {(attention_weights * attention_grads).max():.6f}")
    # print(f"Sum of element-wise products: {(attention_weights * attention_grads).sum():.6f}")

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        scale_factor = 1 #0.0 Scale encoder loss appropriately, if required
       
        tile_loss = (features * gradients).sum()*scale_factor
        

        aloss_innerproduct = (attention_weights * attention_grads).sum()*scale_factor
     
        # Combine losses
        total_loss = tile_loss +  aloss_innerproduct


    # Compute loss and update
    tile_optimizer.zero_grad()
    total_loss.backward()

    tile_optimizer.step()
    
    loss_dict = {'total_loss': total_loss.item(),
                'tile_loss': tile_loss.item(),
                'attention_loss': aloss_innerproduct.item()}
    return loss_dict

def get_cls_attention_weights_features(model, input_tensor, num_heads=24, head_dim=64):
    def hook_fn(module, input, output):
        # QKV computation
        qkv = module.qkv(input[0])
        B, N, _ = qkv.shape
        # print(f"QKV shape: {qkv.shape}")
        
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Compute attention scores
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        # This is the critical fix - compute average attention per image in batch
        # First, get CLS token's attention to all patch tokens
        cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # Average across heads [B, N-1] # change 1: to 5: for h-optimus-0 as it uses 4 registers
        
        # Then sum across patches to get importance score per image
        # This gives us one value per image in the batch
        image_importance = cls_attn.mean(dim=1, keepdim=True)  # [B, 1]
        
        return image_importance
    
        
    # Register hook and storage for result
    attention_result = None
    
    def hook_wrapper(module, input, output):
        nonlocal attention_result
        attention_result = hook_fn(module, input, output)
    
    # Register hook on last block
    last_block = model.blocks[-1]
    hook = last_block.attn.register_forward_hook(hook_wrapper)
    
    # Forward pass
    features = model(input_tensor)
    
    # Remove hook
    hook.remove()
    
    return features, attention_result
    

def main(args):
    print(f"Training with args: {args}")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Setup models and optimizers
    tile_model, tile_optimizer, scaler, tile_scheduler, tsepoch = setup_encoder(args)
    slide_model, slide_optimizer, criterion, slide_scheduler, ssepoch = setup_aggregator(args)
    
    if tsepoch != ssepoch:
        print("Epoch mismatch between encoder and aggregator checkpoints")
        print(f"Encoder: {tsepoch}, Aggregator: {ssepoch}")
        raise ValueError("Epoch mismatch between encoder and aggregator checkpoints")
    else:
        start_epoch = tsepoch
        print(f"Resuming training from epoch {start_epoch}")

    # Create memory-efficient dataset
    dataset = utils_tapfm.SlideSequentialDataset(
        k=args.k_per_gpu,
        splits_csv_path=args.splits_csv_path,
        tile_coords_csv_path=args.tile_coords_csv_path,
        slides_dir_path=args.slides_dir_path,
        tilesize=args.tilesize,
        drop=args.drop,
        target_list=args.target_list,
        rank=0
    )

    # Get one-slide-at-a-time dataloader
    loader = dataset.get_dataloader()


    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
 
    # Training loop
    for epoch in range(start_epoch, args.nepochs + 1):
        print(f"\nEpoch {epoch}/{args.nepochs}")

    
        # Regenerate data for this epoch
        dataset.makeData(epoch)

        running_loss = 0.0
        encoder_loss = 0.0
        
        epoch_predictions = []
        epoch_labels = []
 
        # Process one slide at a time
        for slide_idx, tile_batches in loader:
            # Clear cache at the start of processing each slide
            torch.cuda.empty_cache()
          
            # Get label for this slide
            label = dataset.get_target(slide_idx).to(device)
            storage = utils_tapfm.SharedStorage()  # Reset storage for each slide

            # Process tiles in batches if needed
            all_features = []
            all_attention_weights = []

            for tile_batch in tile_batches:
                # 1. Encoder forward pass
            
                # Get attention weights
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    features, attention_weights = get_cls_attention_weights_features(tile_model, tile_batch.to(args.gpu))
                # print(f"Size of tile batch features: {features.shape} and attention weights:{attention_weights.shape}")
                # print(f"Tile batch features: {features} and attention weights:{attention_weights}")

                all_features.append(features)
                all_attention_weights.append(attention_weights)

            # Concatenate features and attention weights from all batches
            features = torch.cat(all_features, dim=0)
            attention_weights = torch.cat(all_attention_weights, dim=0)

            # Free memory immediately
            del all_features
            del all_attention_weights
            torch.cuda.empty_cache()  # Force CUDA memory cleanup


            # Now process the concatenated attention weights for the whole slide
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                # Apply normalization to entire attention weight vector
                min_attn = attention_weights.min()
                max_attn = attention_weights.max()
                normalized_attn = (attention_weights - min_attn) / (max_attn - min_attn + 1e-8)
                attention_weights = F.softmax(normalized_attn / 0.1, dim=0)

            # print(f"Size of tile features: {features.shape} and attention weights:{attention_weights.shape}")
            # print(f"Tile batch features: {features} and attention weights:{attention_weights}")
            
            # Store in shared storage
            storage.store_features(features)
            storage.store_attention(attention_weights)

            # 2. Aggregator forward/backward pass
            loss, prediction, _ = aggregator_step(
                slide_model, slide_optimizer,
                criterion, storage, dataset, slide_idx, len(loader), args
            )
            running_loss += loss

            # Store predictions
            epoch_predictions.append(prediction.squeeze(0).detach().cpu())
            epoch_labels.append(label.detach().cpu())

            # 3. Encoder backward pass
            eloss = encoder_backward(
                tile_model, tile_optimizer, scaler,
                features, attention_weights, storage, args
            )

            encoder_loss += eloss['total_loss']
            print(f"Slide {slide_idx+1}/{len(loader)} - Aggregator Loss: {loss:.4f} - Encoder Loss: {eloss['total_loss']:.7f}")
        # 
        # Save checkpoints and metrics
        # epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(loader)
        avg_encoder_loss = encoder_loss / len(loader)
        # Calculate AUC
        # Convert lists of tensors to tensors
        epoch_predictions = torch.stack(epoch_predictions).numpy()  # Shape: [num_samples, num_classes]
        epoch_labels = torch.stack(epoch_labels).numpy()            # Shape: [num_samples, num_classes]

        metrics = utils_tapfm.calculate_multilabel_metrics(epoch_predictions, epoch_labels, threshold=None)
        metrics_path = os.path.join(args.outdir, f'metrics_epoch_{epoch}.json')
        # Create the directory if it doesn't exist
        with open(metrics_path, 'w') as f:
            # Simple conversion for numpy types
            json.dump(utils_tapfm.convert_to_serializable(metrics), f, indent=4)

        # Print metrics
        print(f"Epoch {epoch} - Aggregator Loss: {avg_loss:.4f} - Encoder Loss: {avg_encoder_loss:.4f} ")
        print(f"Epoch {epoch} - AUC: {metrics['macro_auc']:.4f} - FPR: {metrics['macro_fpr']:.4f} - FNR: {metrics['macro_fnr']:.4f}")

        utils_tapfm.save_checkpoints(
            epoch, avg_loss, avg_encoder_loss, metrics,
            tile_model, slide_model,tile_optimizer, slide_optimizer, scaler, args)

        tile_scheduler.step()
        slide_scheduler.step()
if __name__ == '__main__':
    # Training Configuration - Modify these as needed
    parser = argparse.ArgumentParser(description='Task Adaptation of Pathology Foundation Model Training')
    
    # Training Configuration
    parser.add_argument('--use_amp', type=int, default=1, help='Use automatic mixed precision (1=True, 0=False)')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer: sgd, adam, adamw')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--nepochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--pfm', type=str, default='gpath', choices=['uni', 'gpath', 'hopt'], 
                       help='Pre-trained foundation model: uni, gpath, or hopt')
    # Data Configuration
    parser.add_argument('--target_list', nargs='+', default=['PIK3CA_Binary', 'FGFR3_Binary'], help='Target classes')
    parser.add_argument('--k_per_gpu', type=int, default=64, help='Tiles per GPU')
    parser.add_argument('--tilesize', type=int, default=224, help='Tile size in pixels')
    parser.add_argument('--drop', type=float, default=0.95, help='Percentage of WSIs that can be dropped during training')
    

    # Data Directories
    parser.add_argument('--splits_csv_path', type=str, 
                       default='./BLCA/Bladder_Urothelial_Carcinoma_manifest_withsplits.csv',
                       help='Path to splits CSV file') # Path to csv file that containts WSI file names and train/test/val split indicators
    parser.add_argument('--tile_coords_csv_path', type=str,
                       default='./BLCA/tile_coords_20x40x.csv', 
                       help='Path to tile coordinates CSV file') # Path to csv file that contains the coordinates from where tiles should be sampled from each WSI
    parser.add_argument('--slides_dir_path', type=str,
                       default='./BLCA',
                       help='Path to slides directory') # path to the directory containing all WSIs

    # Hardware Configuration
    parser.add_argument('--gpu', type=int, default=None, help='GPU device to use (0, 1, 2, etc.)')
    
    # Output Configuration
    parser.add_argument('--outdir', type=str, default='./example_run', help='Output directory')
    parser.add_argument('--save_freq', type=int, default=1, help='Number of epoch after which a model should be saved')
    parser.add_argument('--outname', type=str, default='convergence.csv', help='Output file name')
    # Parse arguments
    args = parser.parse_args()
    
    # Convert use_amp to boolean
    args.use_amp = bool(args.use_amp)
    
    # Set up GPU device properly
    if args.gpu is not None and torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        args.gpu = torch.device(f"cuda:{args.gpu}")
    else:
        args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")


 
    
    main(args)
