# this file is to convert Wav data to numeric data with librosa Audio processing libary
import numpy as np
import pandas as pd
import librosa
import os

location = 'IRMAS-TrainingData/pia'
files_in_dir = []

for file in os.listdir(location):
    if file.endswith(".wav"):
        files_in_dir.append(os.path.join(location, file))

print(len(files_in_dir))


def CAL_Features(file_loc):
    # Load the example clip
    y, sr = librosa.load(file_loc)

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                sr=sr)

    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                        beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

    return beat_features

final_data = [] 
for i in range(len(files_in_dir)):
    song_data = CAL_Features(files_in_dir[i])
    features = []

    features.append(song_data.shape[1])
    features+=(list(song_data.sum(axis=1))) 
    features+=(list(np.mean(song_data,axis=1)))
    features+=(list(np.std(song_data,axis=1)))

    final_data.append(features)

dt = np.array(final_data)
df = pd.DataFrame(dt,columns=range(115))

df.to_csv('piano.csv')
#dt.tofile("electric guitar.csv",sep=',')