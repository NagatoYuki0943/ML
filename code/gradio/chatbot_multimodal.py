# https://www.gradio.app/docs/gradio/chatbot

import gradio as gr
import os
import time

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    print(message)
    # {
    #     'text': 'hello',
    #     'files': [
    #         'C:\\Users\\Frostbite\\AppData\\Local\\Temp\\gradio\\91f58cae35f058d5795dad2a32407d02a0805e10\\MMMMMKUN3 2B-1.jpeg',
    #         'C:\\Users\\Frostbite\\AppData\\Local\\Temp\\gradio\\7b2ef40406377750a86f7b368c760787347f2d9d\\MMMMMKUN3 2B-2.jpeg'
    #     ]
    # }

    for file in message["files"]:
        print(f"file: {file}")
        history.append([(file,), None])
    if message["text"] is not None:
        history.append([message["text"], None])

    print(f"history: {history}")
    # history: [
    #     [('C:\\Users\\Frostbite\\AppData\\Local\\Temp\\gradio\\91f58cae35f058d5795dad2a32407d02a0805e10\\MMMMMKUN3 2B-1.jpeg',), None],
    #     [('C:\\Users\\Frostbite\\AppData\\Local\\Temp\\gradio\\7b2ef40406377750a86f7b368c760787347f2d9d\\MMMMMKUN3 2B-2.jpeg',), None],
    #     ['hello', None]
    # ]

    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot(history):
    response = "**That's cool!**"
    # 添加回答
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.1)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)

    chat_input = gr.MultimodalTextbox(
        file_types=["image"],
        file_count="multiple",  # 指的是一次上传几张,选择single也可以多次选择
        placeholder="Enter message or upload file...",
        label="Prompt",
        interactive=True,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)

demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    max_threads=100,
)
