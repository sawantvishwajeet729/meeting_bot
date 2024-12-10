#import libraries

from resemblyzer import preprocess_wav, VoiceEncoder
import librosa
import numpy as np
from sklearn.cluster import SpectralClustering
import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

#declare variable
#os.environ['gro_key']=st.secrets["openAIKey"]
os.environ['gro_key']="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"
#gro_key = "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"
num_speaker = 2

# Load audio file and preprocess
audio_path = "meeting_sample_1.wav"  # Replace with your file path
wav, sr = librosa.load(audio_path, sr=16000)
wav = preprocess_wav(wav)

# Initialize encoder
encoder = VoiceEncoder()

# Segment and embed
segment_size = 1.5  # in seconds
step_size = 0.75    # in seconds
segments = [
    wav[int(i * sr): int((i + segment_size) * sr)]
    for i in np.arange(0, len(wav) / sr - segment_size, step_size)
]
embeddings = np.array([encoder.embed_utterance(seg) for seg in segments])

# Perform spectral clustering
clustering = SpectralClustering(
    n_clusters=num_speaker, affinity='nearest_neighbors', assign_labels='kmeans'
).fit(embeddings)
labels = clustering.labels_

# Map labels to timestamps
timestamps = []
for i, label in enumerate(labels):
    start_time = i * step_size  # Start time of the segment
    end_time = start_time + segment_size  # End time of the segment
    timestamps.append((start_time, end_time, label))

# Combine consecutive segments with the same speaker
merged_segments = []
current_start, current_end, current_label = timestamps[0]
for start, end, label in timestamps[1:]:
    if label == current_label:
        # Extend the current segment
        current_end = end
    else:
        # Save the previous segment and start a new one
        merged_segments.append((current_start, current_end, current_label))
        current_start, current_end, current_label = start, end, label
# Save the last segment
merged_segments.append((current_start, current_end, current_label))

#initialise whisper model
model = whisper.load_model("base")

#Create transcripts
transcipts = []
for start, end, speaker in merged_segments:
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    split_audio = wav[start_sample:end_sample]
    result_trans = model.transcribe(split_audio)
    transcipts.append(f"speaker {speaker}: {result_trans['text']}")

#combine the list of transcripts into single string
transcipts = '\n'.join(transcipts)

#print(transcipts)

#get groq ouput

system_1 = "you are a skilled editor specializing in enhancing meeting transcripts. Your task is to process raw meeting transcripts to make them clearer, concise, and professional while maintaining the original meaning. Specifically: Clarity: Simplify and rephrase sentences for better readability. Conciseness: Remove filler words, redundancies, and unnecessary phrases. Consistency: Use consistent terminology and tone throughout the transcript. Avoid altering the meaning of the original statements. If the context is unclear, flag such parts for clarification rather than making assumptions. Deliver polished text ready for professional use. Along with this, creata a summary at the end and any action points from the meeting"
human_1 = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system_1), ("human", human_1)])

model = ChatGroq(temperature=0, groq_api_key=gro_key, model_name="llama-3.1-70b-versatile", max_tokens=1024)

output_parser=StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({'text': transcipts})

print(response)