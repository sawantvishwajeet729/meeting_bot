#libraries needed
import streamlit as st
from resemblyzer import preprocess_wav, VoiceEncoder
import librosa
import numpy as np
from sklearn.cluster import SpectralClustering
import whisper
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

#declare variable
#os.environ['gro_key']=st.secrets["openAIKey"]
gro_key = "xxx"

# Initialize encoder
encoder = VoiceEncoder()

#initialise whisper model
model_whisper = whisper.load_model("base")

system_1 = "you are a skilled editor specializing in enhancing meeting transcripts. Your task is to process raw meeting transcripts to make them clearer, concise, and professional while maintaining the original meaning. Specifically: Clarity: Simplify and rephrase sentences for better readability. Conciseness: Remove filler words, redundancies, and unnecessary phrases. Consistency: Use consistent terminology and tone throughout the transcript. Avoid altering the meaning of the original statements. If the context is unclear, flag such parts for clarification rather than making assumptions. Deliver polished text ready for professional use. Along with this, create a short summary at the end along with any action points from the meeting"
human_1 = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system_1), ("human", human_1)])

model = ChatGroq(temperature=0, groq_api_key=gro_key, model_name="llama-3.1-70b-versatile", max_tokens=1024)

output_parser=StrOutputParser()

chain = prompt | model | output_parser

# Segment and embed
segment_size = 1.5  # in seconds
step_size = 0.75    # in seconds

#logo desgin
logo = "images/logo.png"

#---Streamlit code ---
st.set_page_config(page_title="Meeting_Bot", page_icon=":desktop_computer:", layout="wide")

with st.sidebar:
    #---header section---
    with st.container():
        st.image(logo, width=250)
        st.title("Hi there :wave:")
        st.title("Welcome to MeetSync: Simplify Your Meetings")
        st.write("MeetSync is your smart meeting assistant that transforms the way you manage and review meetings. With cutting-edge AI, MeetSync captures meeting transcripts, generates concise summaries, and highlights actionable points. Whether you’re collaborating remotely or in person, MeetSync ensures nothing important is missed, making follow-ups and decision-making effortless.")
        st.write("Stay organized, save time, and keep everyone on the same page with MeetSync—the ultimate tool for efficient and productive meetings.")

with st.container():
    #st.image("logo_white.png", width=400)
    st.title("Welcome to :blue[_MeetSync_]")

with st.container():
    path_url, num_attendee = st.columns((1, 1))
    with path_url:
        audio_path = st.text_input('Enter the path to the meeting audio file')
    with num_attendee:
        num_speaker = st.number_input('Number of Speaker', min_value=1, max_value=10, value=2, step=1)
    
    if st.button("Submit"):
        with st.status("Processing", expanded=False) as status:
            wav, sr = librosa.load(audio_path, sr=16000)
            wav = preprocess_wav(wav)

            segments = [wav[int(i * sr): int((i + segment_size) * sr)]
            for i in np.arange(0, len(wav) / sr - segment_size, step_size)]

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

            #Create transcripts
            transcipts = []
            for start, end, speaker in merged_segments:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                split_audio = wav[start_sample:end_sample]
                result_trans = model_whisper.transcribe(split_audio)
                transcipts.append(f"speaker {speaker}: {result_trans['text']}")

            #combine the list of transcripts into single string
            transcipts = '\n'.join(transcipts)

            st.write("Transcripts created...:white_check_mark:")

            response = chain.invoke({'text': transcipts})
            st.write("Creating Summary...:white_check_mark:")
            
        with st.container():
            st.write(response)

