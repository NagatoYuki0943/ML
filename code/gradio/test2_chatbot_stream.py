# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/gradio/turbomind_coupled.py
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/gradio/vl.py

import gradio as gr
import numpy as np
from typing import Generator, Sequence, Any
import threading
import time
from loguru import logger


logger.info(f"gradio version: {gr.__version__}")


class InterFace:
    global_session_id: int = 0
    lock = threading.Lock()


enable_btn = gr.update(interactive=True)
disable_btn = gr.update(interactive=False)
btn = dict[str, Any]


def chat_stream(
    query: str,
    history: Sequence
    | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    language1: str = "ZH",
    language2: str = "ZH",
    state_session_id: int = 0,
) -> Generator[tuple[Sequence, btn, btn, btn, btn], None, None]:
    history = [] if history is None else list(history)

    logger.info(f"{language1 = }, {language2 = }")
    logger.info(f"{state_session_id = }")
    logger.info(
        {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    )

    if query is None or len(query.strip()) < 1:
        yield history, enable_btn, enable_btn, enable_btn, enable_btn
        return
    query = query.strip()
    logger.info(f"query: {query}")

    yield history + [[query, None]], disable_btn, disable_btn, disable_btn, disable_btn

    time.sleep(1)
    number = np.random.randint(1, 100, 20)
    logger.info(f"response: {number}")
    logger.info(f"history: {history + [[query, str(number)]]}")

    for i in range(len(number)):
        time.sleep(0.1)
        logger.info(number[i])
        yield (
            history + [[query, str(number[: i + 1])]],
            disable_btn,
            disable_btn,
            disable_btn,
            disable_btn,
        )

    yield (
        history + [[query, str(number)]],
        enable_btn,
        enable_btn,
        enable_btn,
        enable_btn,
    )


def regenerate(
    history: Sequence
    | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    language1: str = "ZH",
    language2: str = "ZH",
    state_session_id: int = 0,
) -> Generator[tuple[Sequence, btn, btn, btn, btn], None, None]:
    history = [] if history is None else list(history)

    # 重新生成时要把最后的query和response弹出,重用query
    if len(history) > 0:
        query, _ = history.pop(-1)
        yield from chat_stream(
            query=query,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            language1=language1,
            language2=language2,
            state_session_id=state_session_id,
        )
    else:
        logger.warning("no history, can't regenerate")
        yield history, enable_btn, enable_btn, enable_btn, enable_btn


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
                # 创建聊天框
                chatbot = gr.Chatbot(
                    height=500,
                    show_copy_button=True,
                    placeholder="内容由 AI 大模型生成，请仔细甄别。",
                )

                # 组内的组件没有间距
                with gr.Group():
                    with gr.Row():
                        # 创建一个文本框组件，用于输入 prompt。
                        query = gr.Textbox(
                            lines=1,
                            label="Prompt / 问题",
                            placeholder="Enter 发送; Shift + Enter 换行 / Enter to send; Shift + Enter to wrap",
                        )
                        # 创建提交按钮。
                        # variant https://www.gradio.app/docs/button
                        # scale https://www.gradio.app/guides/controlling-layout
                        submit = gr.Button("💬 Chat", variant="primary", scale=0)

                with gr.Row():
                    # 单选框
                    language1 = gr.Radio(
                        choices=[("中文", "ZH"), ("英文", "EN")],
                        value="ZH",
                        label="Language",
                        type="value",
                        interactive=True,
                    )
                    # 下拉框
                    language2 = gr.Dropdown(
                        choices=[("中文", "ZH"), ("English", "EN")],
                        value="ZH",
                        label="Language",
                        type="value",
                        interactive=True,
                    )
                    # 创建一个重新生成按钮，用于重新生成当前对话内容。
                    regen = gr.Button("🔄 Retry", variant="secondary")
                    undo = gr.Button("↩️ Undo", variant="secondary")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                        components=[chatbot, query], value="🗑️ Clear", variant="stop"
                    )

                # 折叠
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=1,
                            maximum=2048,
                            value=1024,
                            step=1,
                            label="Max new tokens",
                        )
                        temperature = gr.Slider(
                            minimum=0.01,
                            maximum=2,
                            value=0.8,
                            step=0.01,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.01, maximum=1, value=0.8, step=0.01, label="Top_p"
                        )
                        top_k = gr.Slider(
                            minimum=1, maximum=100, value=40, step=1, label="Top_k"
                        )

                gr.Examples(
                    examples=[
                        ["你是谁"],
                        ["你可以帮我做什么"],
                    ],
                    inputs=[query],
                    label="示例问题 / Example questions",
                )

            # 回车提交(无法禁止按钮)
            query.submit(
                chat_stream,
                inputs=[
                    query,
                    chatbot,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                    language1,
                    language2,
                    state_session_id,
                ],
                outputs=[chatbot, submit, regen, undo, clear],
            )

            # 清空query
            query.submit(
                lambda: gr.Textbox(value=""),
                inputs=[],
                outputs=[query],
            )

            # 按钮提交
            submit.click(
                chat_stream,
                inputs=[
                    query,
                    chatbot,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                    language1,
                    language2,
                    state_session_id,
                ],
                outputs=[chatbot, submit, regen, undo, clear],
            )

            # 清空query
            submit.click(
                lambda: gr.Textbox(value=""),
                inputs=[],
                outputs=[query],
            )

            # 重新生成
            regen.click(
                regenerate,
                inputs=[
                    chatbot,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                    language1,
                    language2,
                    state_session_id,
                ],
                outputs=[chatbot, submit, regen, undo, clear],
            )

            # 撤销
            undo.click(revocery, inputs=[chatbot], outputs=[query, chatbot])

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
        max_size=None,  # If None, the queue size will be unlimited.
        default_concurrency_limit=100,  # 最大并发限制
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        max_threads=100,
    )


if __name__ == "__main__":
    main()
