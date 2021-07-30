#!/usr/bin/env python
# coding: utf-8

# In[4]:


import json
import os
import math
import numpy as np
import librosa

mfcc_path = "onemfcc.json"

file_path = "test/reggae/reggae.00001.wav"

SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


# In[5]:


def save_mfcc(file_path, mfcc_path, num_mfcc=13, n_fft=2048, hop_length=512):
    data = {
        "mfcc": []
    }
    
    samples_per_segment = int(SAMPLES_PER_TRACK/10)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    
    signal, sample_rate = librosa.load(file_path, sr=22050)
#     for d in range(1):

    # calculate start and finish sample for current segment
    start = samples_per_segment * 1
    finish = start + samples_per_segment


    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc = 13, n_fft=2048, hop_length=512)

    mfcc = mfcc.T

    mfcc = mfcc[..., np.newaxis]
    print(mfcc.shape)

    if len(mfcc) == num_mfcc_vectors_per_segment:

        data["mfcc"].append(mfcc.tolist())
    
    
    with open(mfcc_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
    
    


# In[6]:


if __name__ == "__main__":
    save_mfcc(file_path, mfcc_path)


# In[ ]:




