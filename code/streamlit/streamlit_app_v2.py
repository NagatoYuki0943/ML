# https://github.com/dataprofessor/llama2/blob/master/streamlit_app_v2.py
# cmd: streamlit run ./streamlit_app_v2.py

import streamlit as st
import numpy as np
import time


print("streamlit version: ", st.__version__)


#----------------------------------------------------------------------#
# prompts (List[str] | str | List[Dict] | List[Dict]): a batch of
#     prompts. It accepts: string prompt, a list of string prompts,
#     a chat history in OpenAI format or a list of chat history.
# [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant."
#     },
#     {
#         "role": "user",
#         "content": "What is the capital of France?"
#     },
#     {
#         "role": "assistant",
#         "content": "The capital of France is Paris."
#     },
#     {
#         "role": "user",
#         "content": "Thanks!"
#     },
#     {
#         "role": "assistant",
#         "content": "You are welcome."
#     }
# ]
#----------------------------------------------------------------------#


# App title
st.set_page_config(page_title="🦙💬 LLaMA 2 Chatbot")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    print("streamlit init messages")
    st.session_state.messages = []


# chat
def chat(
    prompts: list,
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
    temperature: float = 0.8,
    regenerate: bool = False
) -> str:
    """聊天"""
    # 重新生成时要把最后的query和response弹出,重用query
    if regenerate:
        # 有历史就重新生成,没有历史就返回空
        if len(prompts) > 1:
            prompts.pop(-1)
        else:
            return ""

    print(
        {
            "max_new_tokens":  max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature}
    )

    response = str(np.random.randint(1, 100, 10))
    print(response + "\n")
    return response


def regenerate():
    """重新生成"""
    if len(st.session_state.messages) > 0:
        st.session_state.messages.pop(-1)


def undo():
    """恢复到上一轮对话"""
    if len(st.session_state.messages) > 1:
        for i in range(2):
            st.session_state.messages.pop(-1)


# clearn chat history
def clear_chat_history():
    st.session_state.messages = []


def main():
    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 LLaMA 2 Chatbot')
        st.write('This chatbot is created using the open-source LLaMA 2 LLM model from Meta.')

        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a LLaMA2 model', ['LLaMA2-7B', 'LLaMA2-13B', 'LLaMA2-70B'], key='selected_model')

        max_new_tokens = st.sidebar.slider(label='max_new_tokens', min_value=1, max_value=2048, value=1024, step=1)
        top_p = st.sidebar.slider(label='top_p', min_value=0.01, max_value=1.0, value=0.8, step=0.01)
        top_k = st.sidebar.slider(label='top_k', min_value=1, max_value=100, value=40, step=1)
        temperature = st.sidebar.slider(label='temperature', min_value=0.01, max_value=1.5, value=0.8, step=0.01)

        st.subheader('Chat functions')
        st.sidebar.button('🔄 Retry', on_click=regenerate)
        st.sidebar.button('↩️ Undo', on_click=undo)
        st.sidebar.button('🗑️ Clear', on_click=clear_chat_history)

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            # with st.spinner("Thinking..."):
                response = chat(st.session_state.messages, max_new_tokens, top_p, top_k, temperature)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
