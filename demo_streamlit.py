#libraries needed
import streamlit as st

#initialize the logo for webpage rendering
logo = "images/logo.png"

#---Streamlit code ---
st.set_page_config(page_title="Meeting_Bot", page_icon=":desktop_computer:", layout="wide")

#Sidebar Section
with st.sidebar:
    with st.container():
        st.image(logo, width=250)
        st.title("Hi there :wave:")
        st.title("Welcome to MeetSync: Simplify Your Meetings")
        st.write("MeetSync is your smart meeting assistant that transforms the way you manage and review meetings. With cutting-edge AI, MeetSync captures meeting transcripts, generates concise summaries, and highlights actionable points. Whether you’re collaborating remotely or in person, MeetSync ensures nothing important is missed, making follow-ups and decision-making effortless.")
        st.write("Stay organized, save time, and keep everyone on the same page with MeetSync—the ultimate tool for efficient and productive meetings.")

#Main page section: Title
with st.container():
    #st.image("logo_white.png", width=400)
    st.title("Welcome to :blue[_MeetSync_]")

#Main page section: Body
with st.container():
    path_url, num_attendee, groq_key = st.columns((1, 1, 1))
    with path_url:
        #get the path to audio file
        audio_path = st.text_input('Enter the path to the meeting audio file')
    with num_attendee:
        #get the number of speaker in the meeting
        num_speaker = st.number_input('Number of Speaker', min_value=1, max_value=10, value=2, step=1)
    with groq_key:
        #get the GroQ API key
        gro_key = st.text_input('Enter your GroQ API key')
    
    if st.button("Submit"):
        with st.status("Processing", expanded=False) as status:
            st.write("Transcripts created...:white_check_mark:")

            st.write("Creating Summary...:white_check_mark:")
            
        with st.container():
            #print the final result
            st.write("Output of the MeetSync")

