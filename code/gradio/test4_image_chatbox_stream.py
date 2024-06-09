# 导入必要的库
import gradio as gr
import numpy as np
from typing import Generator, Sequence
import threading
import time
from PIL import Image
from loguru import logger
import hashlib


logger.info(f"gradio version: {gr.__version__}")


class InterFace:
    global_session_id: int = 0
    lock = threading.Lock()


def hash_image(image: Image.Image) -> str:
    md5 = hashlib.md5()
    md5.update(image.tobytes())
    return md5.hexdigest()


def chat_stream_with_image(
    query: str,
    history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    image: Image.Image | None = None,
    current_img: str = "",
    state_session_id: int = 0,
) -> Generator[tuple[Sequence, Image.Image], None, None]:
    history = [] if history is None else list(history)

    logger.info(f"{state_session_id = }")

    query = query.strip()
    if query == None or len(query) < 1:
        yield history, current_img
        return

    logger.info({
            "max_new_tokens":  max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
    })

    logger.info(f"{image = }")
    logger.info(f"{current_img = }")
    if isinstance(image, Image.Image):
        new_img_hash = hash_image(image)
        logger.info(f"{new_img_hash = }")

        # 新图片
        if new_img_hash != current_img:
            logger.warning(f"update image hash")
            logger.info({
                "height": image.height,
                "width": image.width,
                "mode": image.mode
            })
            # 转换RGB2BGR
            # image = Image.fromarray(np.array(image)[..., ::-1])
            current_img = new_img_hash
        else:
            # 图片和之前相同设置为 None
            image = None
    else:
        # 不是 PIL.Image.Image 设置为 None
        image = None
    logger.info(f"updated image: {image}")

    logger.info(f"query: {query}")
    number: np.ndarray = np.random.randint(1, 100, 20)
    for i in range(len(number)):
        time.sleep(0.1)
        logger.info(number[i])
        yield history + [[query, str(number[:i+1])]], current_img
        # 在聊天记录中显示图片,需要是图片url,不能是 image 对象
        # yield history + [[("image url",), None], [query, str(number[:i+1])]], image
    logger.info(f"response: {number}")


"""regenerate 和 revocery 在使用图片和文字时会出现问题
假设一下历史记录
    [
        [queston1, answer1], # use image
    ]
    image
在只有一轮对话的情况下,regenerate 需要使用图片和问题, recovery 需要丢弃上一轮的回答和图片

但是在有多轮对话,且上一轮对话没用到图片的情况下
    [
        [queston1, answer1], # not use image
        [queston2, answer2], # use image
    ]
    image
regenerate 不需要使用图片,但是我不知道上一轮是否使用了图片, recovery 需要丢弃上一轮的回答,单不需要丢弃图片
"""

def regenerate(
    query: str,
    history: Sequence | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    image: Image.Image | None = None,
    current_img: str = "",
    state_session_id: int = 0,
) -> Generator[tuple[Sequence, Image.Image], None, None]:
    history = [] if history is None else list(history)

    # 重新生成时要把最后的query和response弹出,重用query
    if len(history) > 0:
        query, _ = history.pop(-1)
        yield from chat_stream_with_image(
            query = query,
            history = history,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            image = image,
            current_img = current_img,
            state_session_id = state_session_id,
        )
    else:
        yield history, image


def revocery(history: Sequence | None = None) -> tuple[str, Sequence]:
    """恢复到上一轮对话"""
    history = [] if history is None else list(history)
    query = ""
    if len(history) > 0:
        query, _ = history.pop(-1)
    return query, history


def main():
    block = gr.Blocks()
    with block as demo:
        state_session_id = gr.State(0)

        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>🦙 LLaMA 3</center></h1>
                    <center>🦙 LLaMA 3 Chatbot 💬</center>
                    """)
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    # 用来存放ocr图片路径，防止重复使用ocr
                    current_img = gr.State("")
                    image = gr.Image(sources=["upload", "webcam", "clipboard"], image_mode="RGB", type="pil", interactive=True)

                    with gr.Column(scale=2):
                        # 创建聊天框
                        chatbot = gr.Chatbot(height=500, show_copy_button=True, placeholder="内容由 AI 大模型生成，请仔细甄别。")

                # 组内的组件没有间距
                with gr.Group():
                    with gr.Row():
                        # 创建一个文本框组件，用于输入 prompt。
                        query = gr.Textbox(
                            lines=1,
                            label="Prompt / 问题",
                            placeholder="Enter 发送; Shift + Enter 换行 / Enter to send; Shift + Enter to wrap"
                        )
                        # 创建提交按钮。
                        # variant https://www.gradio.app/docs/button
                        # scale https://www.gradio.app/guides/controlling-layout
                        submit = gr.Button("💬 Chat", variant="primary", scale=0)

                gr.Examples(
                    examples=[
                        ["你是谁"],
                        ["你可以帮我做什么"],
                    ],
                    inputs=[query],
                    label="示例问题 / Example questions"
                )

                with gr.Row():
                    # 创建一个重新生成按钮，用于重新生成当前对话内容。
                    regen = gr.Button("🔄 Retry", variant="secondary")
                    undo = gr.Button("↩️ Undo", variant="secondary")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(components=[chatbot, image, current_img], value="🗑️ Clear", variant="stop")

                # 折叠
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=1,
                            maximum=2048,
                            value=1024,
                            step=1,
                            label='Max new tokens'
                        )
                        temperature = gr.Slider(
                            minimum=0.01,
                            maximum=2,
                            value=0.8,
                            step=0.01,
                            label='Temperature'
                        )
                        top_p = gr.Slider(
                            minimum=0.01,
                            maximum=1,
                            value=0.8,
                            step=0.01,
                            label='Top_p'
                        )
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=40,
                            step=1,
                            label='Top_k'
                        )

            # 回车提交
            query.submit(
                chat_stream_with_image,
                inputs=[query, chatbot, max_new_tokens, temperature, top_p, top_k, image, current_img, state_session_id],
                outputs=[chatbot, current_img]
            )

            # 清空query
            query.submit(
                lambda: gr.Textbox(value=""),
                [],
                [query],
            )

            # 按钮提交
            submit.click(
                chat_stream_with_image,
                inputs=[query, chatbot, max_new_tokens, temperature, top_p, top_k, image, current_img, state_session_id],
                outputs=[chatbot, current_img]
            )

            # 清空query
            submit.click(
                lambda: gr.Textbox(value=""),
                [],
                [query],
            )

            # 重新生成
            regen.click(
                regenerate,
                inputs=[query, chatbot, max_new_tokens, temperature, top_p, top_k, image, current_img, state_session_id],
                outputs=[chatbot, current_img]
            )

            # 撤销
            undo.click(
                revocery,
                inputs=[chatbot],
                outputs=[query, chatbot]
            )

        gr.Markdown("""提醒：<br>
        1. 内容由 AI 大模型生成，请仔细甄别。<br>
        """)

        # 初始化session_id
        def init():
            with InterFace.lock:
                InterFace.global_session_id += 1
            new_session_id = InterFace.global_session_id
            return new_session_id

        demo.load(init, inputs=None, outputs=[state_session_id])

    # threads to consume the request
    gr.close_all()

    # 设置队列启动
    demo.queue(
        max_size = None,                # If None, the queue size will be unlimited.
        default_concurrency_limit = 100 # 最大并发限制
    )

    # demo.launch(server_name = "127.0.0.1", server_port = 7860, share = True, max_threads = 100)
    demo.launch(max_threads = 100)


if __name__ == "__main__":
    main()
