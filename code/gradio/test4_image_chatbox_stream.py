# 导入必要的库
import gradio as gr
import numpy as np
from typing import Generator, Any
import time
from PIL import Image


print("gradio version: ", gr.__version__)


def chat_stream_with_image(
    query: str,
    history: list | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    image: Image.Image | None = None,
) -> Generator[Any, Any, Any]:
    history = [] if history is None else history

    query = query.strip()
    if query == None or len(query) < 1:
        yield history, image
        return

    print({
            "max_new_tokens":  max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
    })

    if isinstance(image, Image.Image):
        print({
            "height": image.height,
            "width": image.width,
            "mode": image.mode
        })
        # 转换RGB2BGR
        image = Image.fromarray(np.array(image)[..., ::-1])

    print(f"query: {query}; response: ", end="", flush=True)
    number = np.random.randint(1, 100, 10)
    for i in range(10):
        time.sleep(0.1)
        print(number[i], end=" ", flush=True)
        yield history + [[query, str(number[:i+1])]], image
    print("\n")


def regenerate(
    query: str,
    history: list | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    image: Image.Image | None = None,
) -> Generator[Any, Any, Any]:
    history = [] if history is None else history

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
        )
    else:
        yield history, image


def revocery(history: list | None) -> tuple[str, list]:
    """恢复到上一轮对话"""
    history = [] if history is None else history
    query = ""
    if len(history) > 0:
        query, _ = history.pop(-1)
    return query, history


def main():
    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>🦙 LLaMA 3</center></h1>
                    <center>🦙 LLaMA 3 Chatbot 💬</center>
                    """)
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    image = gr.Image(image_mode="RGB", type="pil", interactive=True)

                    with gr.Column(scale=2):
                        # 创建聊天框
                        chatbot = gr.Chatbot(height=500, show_copy_button=True, placeholder="内容由 AI 大模型生成，不构成专业医疗意见或诊断。")

                        with gr.Row():
                            # 创建一个文本框组件，用于输入 prompt。
                            query = gr.Textbox(label="Prompt/问题", placeholder="Enter 发送; Shift + Enter 换行 / Enter to send; Shift + Enter to wrap")
                            # 创建提交按钮。
                            # variant https://www.gradio.app/docs/button
                            # scale https://www.gradio.app/guides/controlling-layout
                            submit = gr.Button("💬 Chat", variant="primary", scale=0)

                with gr.Row():
                    # 创建一个重新生成按钮，用于重新生成当前对话内容。
                    regen = gr.Button("🔄 Retry", variant="secondary")
                    undo = gr.Button("↩️ Undo", variant="secondary")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(components=[chatbot], value="🗑️ Clear", variant="stop")

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
                            maximum=1.5,
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
                inputs=[query, chatbot, max_new_tokens, temperature, top_p, top_k, image],
                outputs=[chatbot, image]
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
                inputs=[query, chatbot, max_new_tokens, temperature, top_p, top_k, image],
                outputs=[chatbot, image]
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
                inputs=[query, chatbot, max_new_tokens, temperature, top_p, top_k, image],
                outputs=[chatbot, image]
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

    # threads to consume the request
    gr.close_all()

    # 设置队列启动，队列最大长度为 100
    demo.queue(max_size=100)

    # demo.launch(server_name="127.0.0.1", server_port=7860)
    demo.launch()


if __name__ == "__main__":
    main()
