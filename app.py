import streamlit as st
from pathlib import Path
import time
from backend_code import chatbot, vector_db_creation # Ensure this import is correct and matches your backend file and function
from langchain.memory import ConversationBufferMemory

WELCOME_STR = "Hello! I am EngiGuide, your companion to answer any queries you may have about the FAMU FSU COE. Start typing away, and I'll be happy to assist!"
LOCAL_VECTOR_STORE_DIR = (Path(__file__).resolve().parent.joinpath("data", "vector_stores"))

def response_generator():
    response = WELCOME_STR
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def main():
    st.set_page_config(page_title="FAMU FSU College of Engineering")
    st.image(image="https://eng.famu.fsu.edu/sites/g/files/upcbnu1751/files/images/marcom/F2COE-ECE.png")

    st.header("EngiGuide: FAMU FSU COE Guide for Undergraduate Students")

    st.sidebar.title("About the app:")
    st.sidebar.info("""
    **EngiGuide** is a Generative AI chat assistant powered by Google Gemini and Langchain, that is designed to help undergraduate students navigate the FAMU-FSU College of Engineering. Whether you have questions about courses, faculty, campus resources, or need guidance on academic policies, EngiGuide is here to assist you 24/7. Just type your query, and EngiGuide will provide the information you need, sourced from the FAMU-FSU Undergraduate Handbook. Start exploring and let EngiGuide enhance your college experience!
    """)
        

    if "db" not in st.session_state:
        st.session_state.db = vector_db_creation()
    
    st.write_stream(response_generator())

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", input_key="question",return_messages=True)

    if prompt := st.chat_input("Type your query"):
        st.chat_message("user").write(prompt)

        # Call the backend function to get the assistant's response
        response, st.session_state.memory = chatbot(prompt,st.session_state.db,st.session_state.memory)  

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        
        with st.chat_message("assistant"):
            st.markdown(response)
        


if __name__ == "__main__":
    main()