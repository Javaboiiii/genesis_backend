from transformers import pipeline, AutoModel, AutoFeatureExtractor
import torch
import os 
import time 

# Global variables to cache model and feature extractor
_feature_extractor = None
_model = None
model_ckpt = "m-a-p/music2vec-v1"

def load_model():
    """Load model and feature extractor only when needed (lazy loading)"""
    global _feature_extractor, _model
    if _feature_extractor is None or _model is None:
        print("Loading Music2Vec model and feature extractor...")
        start_time = time.time()
        _feature_extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        _model = AutoModel.from_pretrained(model_ckpt)
        load_time = time.time() - start_time
        print(f"Model loading time: {load_time:.2f} seconds")
    return _feature_extractor, _model


def extract_music_features(audio_file_path):
    """Extract features from audio file"""
    try:
        # Load model and feature extractor (cached after first call)
        feature_extractor, model = load_model()
        
        import librosa
        audio, sr = librosa.load(audio_file_path, sr=16000)

        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            features = model(**inputs).last_hidden_state
        
        return features.mean(dim=1).squeeze().tolist()
    
    except Exception as e:
        print(f"Error processing audio {audio_file_path}: {str(e)}")
        return None  


if __name__ == "__main__":
    print("=== Music2Vec Audio Embedder Test ===")
    
    # For audio/music processing
    filename = "song2.mp3"  
    music_file_path = os.path.join("musics", filename)
    print(f"Processing: {music_file_path}")
    
    # Time the entire process
    total_start = time.time()
    
    # First embedding (includes model loading)
    print("\n--- First run (includes model loading) ---")
    start = time.time()
    result1 = extract_music_features(music_file_path)
    stop = time.time()
    first_run_time = stop - start
    
    if result1:
        print(f"Embedding dimension: {len(result1)}")
        print(f"First 5 values: {result1[:5]}")
        print(f'First run processing time: {first_run_time:.2f} seconds')
    
    # Second embedding (model already loaded)
    print("\n--- Second run (model cached) ---")
    start = time.time()
    result2 = extract_music_features(music_file_path)
    stop = time.time()
    second_run_time = stop - start
    
    if result2:
        print(f'Second run processing time: {second_run_time:.2f} seconds')
        print(f'Speed improvement: {first_run_time/second_run_time:.1f}x faster')
    
    total_time = time.time() - total_start
    print(f'\nTotal execution time: {total_time:.2f} seconds')
    
    if not result1:
        print("Failed to extract features")

