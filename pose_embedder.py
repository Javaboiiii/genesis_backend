from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image
import os
import time

# Global variables to cache model and processor
_processor = None
_model = None
model_ckpt = "nateraw/vit-base-beans"

def load_model():
    """Load model and processor only when needed (lazy loading)"""
    global _processor, _model
    if _processor is None or _model is None:
        print("Loading ViT model and processor...")
        start_time = time.time()
        _processor = ViTImageProcessor.from_pretrained(model_ckpt)
        _model = ViTModel.from_pretrained(model_ckpt)
        load_time = time.time() - start_time
        print(f"Model loading time: {load_time:.2f} seconds")
    return _processor, _model


def pose_embedder(file_path):
    """Extract features from image file for pose/image embedding"""
    try:
        # Load model and processor (cached after first call)
        processor, model = load_model()
        
        # Load and process the image
        image = Image.open(file_path).convert('RGB')
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the pooler output for a single embedding vector
            # If pooler is not available, use the mean of last_hidden_state
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                # Fall back to mean pooling of the patch embeddings
                features = outputs.last_hidden_state.mean(dim=1)
        
        return features.squeeze().tolist()
    
    except Exception as e:
        print(f"Error processing image {file_path}: {str(e)}")
        return None


if __name__ == "__main__":
    print("=== ViT Image Embedder Test ===")
    
    path = os.getcwd() 
    image_name = "ash_greNinja.jpg"
    pose_file = path + f'/images/{image_name}' 
    print(f"Processing: {pose_file}")
    
    # Time the entire process
    total_start = time.time()
    
    # First embedding (includes model loading)
    print("\n--- First run (includes model loading) ---")
    start = time.time()
    result1 = pose_embedder(pose_file)
    stop = time.time()
    first_run_time = stop - start
    
    if result1:
        print(f"Embedding dimension: {len(result1)}")
        print(f"First 5 values: {result1[:5]}")
        print(f'First run processing time: {first_run_time:.2f} seconds')
    
    # Second embedding (model already loaded)
    print("\n--- Second run (model cached) ---")
    start = time.time()
    result2 = pose_embedder(pose_file)
    stop = time.time()
    second_run_time = stop - start
    
    if result2:
        print(f'Second run processing time: {second_run_time:.2f} seconds')
        print(f'Speed improvement: {first_run_time/second_run_time:.1f}x faster')
    
    total_time = time.time() - total_start
    print(f'\nTotal execution time: {total_time:.2f} seconds')
    
    if not result1:
        print("Failed to extract features")