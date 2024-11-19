import streamlit as st
from datetime import datetime
from inference import *
import pandas as pd

# Function to style Streamlit app
def style_app():
    st.markdown("""<style>
        .chat-container { padding: 10px; margin: 0; border-radius: 10px; background-color: #f7f7f7; font-size: 16px; color: #00000; }
        .user-message { background-color: #d9f7c4; color: black; padding: 10px; border-radius: 10px; margin-bottom: 5px; text-align: left; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); }
        .bot-message { background-color: #e0f7fa; color: black; padding: 10px; border-radius: 10px; margin-bottom: 5px; text-align: left; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); }
        .header { background-color: #4CAF50; color: black; padding: 15px; text-align: center; font-size: 24px; border-radius: 10px; margin-bottom: 20px; }
        .subheader { font-size: 20px; font-weight: bold; color: #00796b; }
        .footer { font-size: 14px; color: #00796b; text-align: center; margin-top: 30px; }
        .profile-container { display: flex; align-items: center; margin: 20px 0; }
        .profile-image { width: 60px; height: 60px; border-radius: 50%; background-color: #e0f7fa; margin-right: 10px; }
        .note-container { margin-top: 20px; border: 1px solid #00796b; padding: 10px; border-radius: 5px; background-color: #f1f1f1; }
        .logout-btn { position: absolute; top: 20px; right: 20px; background-color: #f44336; color: white; padding: 10px; border-radius: 50%; cursor: pointer; font-size: 16px; }
        .logout-btn:hover { background-color: #d32f2f; }
    </style>""", unsafe_allow_html=True)

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# Users DB simulation
users_db = {}

# Registration and Login
def register_user(username, password, full_name, dob, email):
    st.session_state.user_profile = {
        "full_name": full_name,
        "dob": dob,
        "email": email,
        "username": username,
        "password": password,
        "chat_history": [] 
    }
    users_db[username] = st.session_state.user_profile 
    st.session_state.chat_history = []

def authenticate_user(username, password):
    if username in users_db and users_db[username]["password"] == password:
        return True
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.chat_history = []
    st.session_state.user_profile = {}

def main():
    style_app() 
    st.markdown("<div class='header'>Mental Health Chatbot</div>", unsafe_allow_html=True)

    if st.session_state.logged_in:
        if st.button('Logout', key='logout_button', use_container_width=True):
            logout()
            st.write("You have logged out successfully.")
            return

    if not st.session_state.logged_in:
        st.subheader("Please log in to start chatting:")
        login_option = st.selectbox("Choose an option", ["Login", "Register", "Quit"])

        if login_option == "Login":
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    st.session_state.chat_history = users_db[username]["chat_history"]
                else:
                    st.error("Invalid credentials.")

        elif login_option == "Register":
            full_name = st.text_input("Full Name", key="full_name")
            dob = st.date_input("Date of Birth", min_value=datetime(1950, 1, 1), max_value=datetime.today())
            email = st.text_input("Email ID", key="email")
            username = st.text_input("Choose a Username", key="register_username")
            password = st.text_input("Choose a Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if password != confirm_password:
                st.error("Passwords do not match!")

            if st.button("Register"):
                if password == confirm_password:
                    register_user(username, password, full_name, dob, email)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Registration successful!")

        elif login_option == "Quit":
            st.write("You chose to quit the app.")
            return

    if st.session_state.logged_in:
        st.subheader(f"Hello, {st.session_state.username}! Welcome to your profile.")
        
        option = st.sidebar.radio("Choose an Option", ["Chat", "Profile", "Chat History"])

        if option == "Chat":
            for message in st.session_state.chat_history:
                if message.startswith("You:"):
                    st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
                elif message.startswith("Bot:"):
                    st.markdown(f"<div class='bot-message'>{message}</div>", unsafe_allow_html=True)

            with st.form(key='chat_form', clear_on_submit=True):
                user_message = st.text_input("You:", key="user_input", label_visibility="collapsed")
                submit_button = st.form_submit_button("Send")
                
                tokenizer_file_path = r"D:\Intership\Infosys Springboard\tokenizer.pkl"
                dataframe_File_path = r"D:\Intership\Infosys Springboard\main_df.csv"
                df = pd.read_csv(dataframe_File_path)
                encoder_File_path = r"D:\Intership\Infosys Springboard\label_encoder.pkl"
                model_file_path = r"D:\Intership\Infosys Springboard\my_model.h5"
                tokenizer, encoder = load_tokenizer_and_encoder(tokenizer_File_path=tokenizer_file_path, encoder_file_path=encoder_File_path)
                model = load_model(model_file_path)

                if submit_button and user_message:
                    st.session_state.chat_history.append(f"You: {user_message}")
                    response = generate_answer(pattern=f"{user_message}", tokenizer=tokenizer, df=df, lbl_enc=encoder, model=model)
                    st.session_state.chat_history.append(f"Bot: {response}")
                    st.write(response)

        elif option == "Profile":
            if st.session_state.user_profile:
                st.markdown(f"<div class='profile-container'><div class='profile-image'></div><div><b>Full Name:</b> {st.session_state.user_profile['full_name']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div><b>Date of Birth:</b> {st.session_state.user_profile['dob']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div><b>Email ID:</b> {st.session_state.user_profile['email']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div><b>Username:</b> {st.session_state.user_profile['username']}</div>", unsafe_allow_html=True)
                
                note = st.text_area("Write a note for yourself:")
                if st.button("Save Note"):
                    st.session_state.user_profile["note"] = note
                    st.success("Note saved!")

                if "note" in st.session_state.user_profile:
                    st.markdown(f"<div class='note-container'><b>Your Note:</b><br>{st.session_state.user_profile['note']}</div>", unsafe_allow_html=True)

        elif option == "Chat History":
            st.subheader("Your Chat History")
            for message in st.session_state.chat_history:
                st.write(message)

if __name__ == "__main__":
    main()

