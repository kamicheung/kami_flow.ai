import time

import openai
import streamlit as st

from src.agent import load_agent
from src.auto_apply import auto_apply_bot
from src.utils import parse_resume_upload, save_user_info

openai.api_key = st.secrets['OPENAI_API_KEY']

st.set_page_config(
    page_title='This is a test',
    page_icon='💀'
)

custom_css = """
<style>
    .stApp {
        background: linear-gradient(to bottom, #99ccff 0%, #ffffff 90%);
    }
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)
st.title('FlowX ai')
st.text("Your personal career assistant")

# Initiliaze the message history and role display
if 'message' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

with st.sidebar:
    desired_job_title = st.text_input("Desired Job Title", "Data Scientist")
    mb_type = st.selectbox("What's your Myers-Briggs indicator?",(
        "Unsure", "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ",
    ),

    )
    st.markdown(
        "[Unsure? Take the Personality Test](https://www.16personalities.com/free-personality-test)"        
    )
    user_interests = st.multiselect("What are you interests? Select all that apply.",
                                    [
                                        "Academia", "Arts", "Business", "Computing", "Cooking", "Design",
                                        "Economics", "Engineering", "Finance", "Fitness", "Healthcare",
                                        "Law", "Marketing", "Media", "Politics", "Psychology", "Science",
                                        "Sports", "Sustainability", "Technology", "Writing",
                                    ],
                                    ["Academia"],
    )
    uploaded_resume = st.file_uploader(
        "Upload you resume", type=['pdf','doc','txt']
    )
    if uploaded_resume is not None:
        parse_resume_upload(uploaded_resume)

    if st.button("Load Assistant"):
        save_user_info(desired_job_title, mb_type, user_interests)
        agent = load_agent(desired_job_title, mb_type, user_interests)
        st.session_state['agent'] = agent
        st.success("Assistant Loaded!", "💀💀💀")

    st.divider()
    st.subheader("💀💀💀 :violet[PRO] 💀💀💀")
    if st.button["Auto-Apply"]:
        st.info("Auto Apply started!", icon="💀")
        auto_apply_bot(desired_job_title)

    st.divider()
    st.header("Instructions:")
    st.write( "1. Input your preferred job title, Myers-Briggs Type Indicator, interests, and upload your CV.")
    st.write("2. Initiate your personalized career AI assistant by clicking 'Load Assistant'.")
    st.write("3. Our system, Flow.ai, will then scan Google Jobs to extract job postings matching your desired job title.")
    st.write("4. Feel free to ask questions related to your CV or preferred job title.")
    st.write("5. Flow.ai will also propose job opportunities tailored to your resume and the job title you're interested in.")
    st.write("6. Beyond this, Flow.ai can provide you with targeted advice gleaned from a vast array of internet sources.")

# Create the prompt system
# Code adapted from:
# https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

# React to user input
if prompt := st.chat_input("Let's start your career journey!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner('hmmmmmmm..'):
            assistant_repsonse = st.session_state['agent'].run(prompt)
        st.markdown(assistant_repsonse)
        full_response = assistant_repsonse
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})