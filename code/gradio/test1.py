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
demo.launch()
