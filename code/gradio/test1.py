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
# 设置队列启动
demo.queue(max_size=100, concurrency_count=10)

demo.launch(
    server_name = "0.0.0.0",
    server_port = 7860,
    share = True,
    max_threads = 100,
)
