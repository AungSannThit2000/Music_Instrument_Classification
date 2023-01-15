import streamlit as st
import pandas as pd
from io import StringIO

import numpy as np

st.write("THis is music insturement sound classifier")


uploaded_files = st.file_uploader("Choose a Wav file", type = 'wav', accept_multiple_files= True)
for uploaded_file in uploaded_files:
    wav_file_upload = uploaded_file.read()
    st.write("filename:", wav_file_upload)


    audio, sample_rate = librosa.load(wav_file_upload, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    st.write(mfccs_scaled_features)
