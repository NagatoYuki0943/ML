import gradio as gr
import numpy as np


history = []


def chat(query):
    global history
    history.append((query, str(np.random.randint(1, 100, 10))))
    return history


textbox = gr.Textbox()
chatbox = gr.Chatbot()

demo = gr.Interface(fn=chat, inputs=[textbox], outputs=[chatbox])
# 设置队列启动，队列最大长度为 100
demo.queue(max_size=100)
demo.launch(server_name="127.0.0.1", server_port=7860)
