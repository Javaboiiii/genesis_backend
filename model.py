from transformers import pipeline, AutoModel, AutoFeatureExtractor
import torch
import os 
import time 

filename = "song2.mp3"  
music_file_path = os.path.join("musics", filename)  

# # For audio/music processing
feature_extractor = AutoFeatureExtractor.from_pretrained("m-a-p/music2vec-v1")
model = AutoModel.from_pretrained("m-a-p/music2vec-v1")

def extract_music_features(audio_file_path):
    """Extract features from audio file"""
    import librosa
    audio, sr = librosa.load(audio_file_path, sr=16000)

    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        features = model(**inputs).last_hidden_state
    
    return features.mean(dim=1).squeeze().tolist()  


start = time.time()
result = extract_music_features(music_file_path)
stop = time.time()

print(result)
print(f'Processing time: {stop - start:.2f} seconds')

