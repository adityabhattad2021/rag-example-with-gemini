import streamlit as st
from chat import generate_reply as get_response

st.title("ChatMed")

if "messages" not in st.session_state:
  st.session_state.messages = []

# For displaying message from history on rerun
for message in st.session_state.messages:
  with st.chat_message(message['role']):
    st.markdown(message['content'])


if prompt := st.chat_input("Message..."):
  with st.chat_message("user"):
    st.markdown(prompt)

  st.session_state.messages.append({"role": "user", "content": prompt})

  response = get_response(prompt)

  with st.chat_message("assistant"):
    st.markdown(response)

  st.session_state.messages.append({"role": "assistant", "content": response})
