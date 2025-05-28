import torch
import os
import pandas as pd
import sys
import numpy as np
import pandas as pd
from SlideTileExtractor import extract_tissue
import openslide 
import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import utils_tapfm
import torch.nn.functional as F
import glob
import csv
import pandas as pd
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import time

# Define transform pipeline for tiles
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

def main():
    mutation_list = ['PIK3CA_Binary', 'FGFR3_Binary']
    # svs and ground truth files path
    # BLCA paths
    master_df = pd.read_csv('./BLCA/BLCA_combined_mutations.csv') # splits file that contains wsi filename
    results_dir = './BLCA/example_run/' # directory where results will be saved
    model_dir = './BLCA/full_run/' # directory from where to load the trained model


    tile_checkpoint_path =  os.path.join(model_dir, 'checkpoint_tile_011.pth')
    slide_checkpoint_path = os.path.join(model_dir, 'checkpoint_slide_011.pth')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    results_filename = os.path.join(model_dir, 'validation_results_20x40x.csv')

    master_df['Slide_Resolution'] = None
    master_df['MAXMPP_Resolution'] = None

    # Automatically create prediction columns for each mutation
    for mutation in mutation_list:
        # Extract the gene name (remove "_Binary" suffix)
        gene_name = mutation.split('_')[0]
        # Create prediction column
        master_df[f'{gene_name}_prediction_prob'] = None

    # Initialize the encoder model
    tile_model = utils_tapfm.get_pfm('gpath')
     # get head info for the encoder (PFM) model
    num_heads, head_dim, hidden_dim = utils_tapfm.get_head_info(tile_model)

    tile_model.ndim = hidden_dim  # Set the feature dimension

    
    tile_model = utils_tapfm.load_checkpoint(tile_checkpoint_path, tile_model, device, model_type='tile')
    tile_model = tile_model.to(device)
    tile_model.eval()
    print("Tile Model Loaded.")
    

    slide_model = utils_tapfm.tapfm_aggregator(ndim=tile_model.ndim, n_classes=len(mutation_list))
    slide_model = utils_tapfm.load_checkpoint(slide_checkpoint_path, slide_model, device, model_type='slide')
    slide_model = slide_model.to(device)
    slide_model.eval()
    print("Slide Model Loaded.")

    # Initialize lists to store target and predicted values
    all_targets = []
    all_predictions = []

    validation_start_time = time.time()
    validation_times = []
    timing_csv_path = os.path.join(results_dir, 'validation_timing.csv')

    # Loop through the slides in the manifest
    for i, row in master_df.iterrows():
        wsi_start_time = time.time()
        print(f"Processing Image {i+1}/{len(master_df)}")
        slide_path = row.svs_path

        # Extract ground truth labels for the specific mutations
        ground_truth = []
        for mutation in mutation_list:  # Where mutation_list = ['PIK3CA_Binary', 'FGFR3_Binary'] for BLCA
            if pd.isna(row[mutation]):
                # Skip slides with missing ground truth
                print(f"Skipping sample {i+1} - Missing ground truth for {mutation}")
                continue
            ground_truth.append(int(row[mutation]))
        all_targets.append(ground_truth)

        if pd.isna(slide_path):
            print(f"Skipping sample {i+1} - No SVS path found")
            continue
            
        # print(f"Loading slide from {slide_path}")

        try:
            # Open the slide
            slide = openslide.OpenSlide(slide_path)
            # Extract slide metadata
            resolution = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            mpp_resolution = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            master_df.at[i, 'Slide_Resolution'] = resolution
            master_df.at[i, 'MAXMPP_Resolution'] = mpp_resolution
            
            # Determine patch size based on resolution
            psize = 224  # Default size
            if mpp_resolution >= 0.45:
                psize = 224
            elif mpp_resolution < 0.30 and mpp_resolution >= 0.0001:
                psize = 448  # Use 448 for processing 40x WSIs with 20x model
            
            # Extract all tile coordinates
            tilecoords_to_load = pd.DataFrame(
                extract_tissue.make_sample_grid(
                    slide, 
                    patch_size=psize, 
                    mpp=mpp_resolution, 
                    overlap=1, 
                    mult=4
                ), 
                columns=['x', 'y']
            )
            print(f"    Found {len(tilecoords_to_load)} tiles for processing")

            
            # Process tile coordinates in chunks with chunk size of 1000
            chunk_size = 1000
            num_chunks = (len(tilecoords_to_load) + chunk_size - 1) // chunk_size

            # Initialize feature and attention containers
            all_features = []
            all_attention_weights = []

            # for chunk_idx in range(num_chunks):
            for chunk_idx in tqdm(range(num_chunks), desc="Processing Chunks", position=1, leave=False):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(tilecoords_to_load))
    
                # print(f"   Processing chunk {chunk_idx+1}/{num_chunks} (tiles {start_idx} to {end_idx-1})")
    
                # Extract tiles for this chunk
                chunk_tiles = []
                for idx in range(start_idx, end_idx):
                    x, y = tilecoords_to_load.iloc[idx]['x'], tilecoords_to_load.iloc[idx]['y']
        
                    # Extract the tile
                    img = Image.fromarray(np.array(slide.read_region(
                    location=(int(x), int(y)),
                    size=(psize, psize),
                    level=0
                    ))).convert('RGB')
        
                    # Resize if psize is larger than 224
                    if psize > 224:
                        img = img.resize((224, 224), Image.LANCZOS)


                    # sample_tiles_dir = os.path.join(results_dir, 'sample_tiles')
                    # os.makedirs(sample_tiles_dir, exist_ok=True)
                    # tile_save_path = os.path.join(sample_tiles_dir, f'slide_{i}_tile_{idx}.png')
                    # img.save(tile_save_path)

                    # Apply transform
                    img_tensor = transform(img)
                    chunk_tiles.append(img_tensor)
    
                # Stack tiles into a batch tensor
                inputs = torch.stack(chunk_tiles).to(device)
                # print(f"Size of input tensors {inputs.size()}")
                # Process the chunk
                with torch.no_grad():
                    chunk_features = utils_tapfm.encoder_forward(tile_model, inputs)
                    # print(f"Feature shape {chunk_features}")
                    chunk_attention_weights, _ = utils_tapfm.get_cls_attention_weights_ddp(tile_model, inputs)
    
                # Append to lists
                all_features.append(chunk_features.cpu())
                all_attention_weights.append(chunk_attention_weights.cpu())

            # Concatenate all features and attention weights
            all_features = torch.cat(all_features, dim=0).to(device)
            all_attention_weights = torch.cat(all_attention_weights, dim=0).to(device)

            # Pass the features matrix and attention vector through aggregator
            with torch.no_grad():
                _, output = slide_model(all_features, all_attention_weights)
            torch.cuda.synchronize()  # Synchronize after forward pass

            # Get predictions for all mutations
            predictions = torch.sigmoid(output).cpu().numpy()

            # End timer for this WSI
            wsi_end_time = time.time()
            wsi_processing_time = wsi_end_time - wsi_start_time
            validation_times.append(wsi_processing_time)

            # for BLCA
            master_df.at[i, 'PIK3CA_prediction_prob'] = predictions[0, 0]
            master_df.at[i, 'FGFR3_prediction_prob'] = predictions[0, 1]

            # Store predictions for metrics calculation
            all_predictions.append(predictions[0])
            # Compute AUC for each mutation
        
            # break
        except Exception as e:
            print(f"Error processing slide {i+1} ({slide_path}): {str(e)}")
            # Log the error
            error_type = type(e).__name__
            master_df.at[i, 'Processing_Error'] = f"{error_type}: {str(e)}"
            # Continue to the next slide
            continue
        # break
    epoch = 11
    validation_end_time = time.time()
    total_validation_time = validation_end_time - validation_start_time
    avg_time_per_wsi = np.mean(validation_times)
    std_time_per_wsi = np.std(validation_times)
    throughput = len(validation_times) / total_validation_time
    timing_data = {
                    'total_time': total_validation_time,
                    'avg_time_per_wsi': np.mean(validation_times),
                    'std_time_per_wsi': np.std(validation_times),
                    'min_time': np.min(validation_times),
                    'max_time': np.max(validation_times),
                    'throughput': len(validation_times) / total_validation_time
                    }
    utils_tapfm.save_timing_to_csv(epoch, timing_data, timing_csv_path)

    # Save the dataframe with predictions
    predictions_file_path = os.path.join(results_dir, 'BLCA_predictions.csv')
    master_df.to_csv(predictions_file_path, index=False)
    print(f"Results saved to BLCA_predictions.csv")

        # Convert lists to numpy arrays for metrics calculation
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    print(f"Collected predictions for {len(all_predictions)} slides")
    print(f"Target shape: {all_targets.shape}, Predictions shape: {all_predictions.shape}")
    
    # Calculate metrics
    metrics = utils_tapfm.calculate_multilabel_metrics(all_predictions, all_targets)
    # Save metrics to JSON file
    metrics_file_path = os.path.join(results_dir, 'validation_metrics.json')
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f, indent=4, cls=utils_tapfm.NumpyEncoder)
    
    print(f"Metrics saved to {metrics_file_path}")
    
               
if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")   
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    main()
    print("Processing completed successfully!")
    # Save the final results
   
    print(f"Results saved to BLCA_predictions.csv")