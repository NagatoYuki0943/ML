import time
import gradio as gr


def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.05)
        yield "You typed: " + message[: i + 1]


demo = gr.ChatInterface(slow_echo).queue()

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    max_threads=100,
)
