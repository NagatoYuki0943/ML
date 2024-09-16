"""https://www.gradio.app/docs/gradio/tab_summary_"""

import numpy as np
import gradio as gr
from loguru import logger


logger.info(f"gradio version: {gr.__version__}")


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")

    # tab代表选项卡
    with gr.Tab("Flip Text"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")

    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")

    with gr.Accordion("Open for More!", open=False):
        gr.Markdown("Look at me...")
        temp_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.1,
            step=0.1,
            interactive=True,
            label="Slide me",
        )
        temp_slider.change(lambda x: x, [temp_slider])

    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    max_threads=100,
)
