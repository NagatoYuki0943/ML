# https://www.gradio.app/playground

import random
import gradio as gr

def random_response(message, history):
    return random.choice(["Yes", "No"])

demo = gr.ChatInterface(random_response)

demo.launch(
    server_name = "0.0.0.0",
    server_port = 7860,
    share = True,
    max_threads = 100,
)
