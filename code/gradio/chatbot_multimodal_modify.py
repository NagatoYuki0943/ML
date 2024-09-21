# 导入必要的库
import gradio as gr
import numpy as np
from typing import Sequence
import threading
import time
from loguru import logger
from pathlib import Path


logger.info(f"gradio version: {gr.__version__}")
save_path = Path("upload")
save_path.mkdir(parents=True, exist_ok=True)


class InterFace:
    global_session_id: int = 0
    lock = threading.Lock()


def multimodal_chat(
    query: dict,
    history: Sequence
    | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    state_session_id: int = 0,
) -> Sequence:
    history = [] if history is None else list(history)

    logger.info(f"{state_session_id = }")
    logger.info(
        {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    )

    logger.info(f"query : {query }")
    query_text = query["text"]
    # if query_text is None or len(query_text.strip()) == 0:
    if query_text is None or (
        len(query_text.strip()) == 0 and len(query["files"]) == 0
    ):
        logger.warning("query is None, return history")
        return history
    query_text = query_text.strip()
    logger.info(f"query_text: {query_text}")

    # 将图片放入历史记录中
    for file in query["files"]:
        logger.info(f"{file = }")
        history.append([(file,), None])

    time.sleep(3)
    response = str(object=np.random.randint(1, 100, 20))
    logger.info(f"response: {response}")
    history.append([query_text, response])
    logger.info(f"history: {history}")

    return history


def regenerate(
    history: Sequence
    | None = None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    state_session_id: int = 0,
) -> Sequence:
    history = [] if history is None else list(history)

    query = {"text": "", "files": []}
    # 重新生成时要把最后的query和response弹出,重用query
    if len(history) > 0:
        query_data, _ = history.pop(-1)
        if isinstance(query_data, str):
            query["text"] = query_data
        else:
            # 获取文件
            query["files"].append(query_data[0])
        return multimodal_chat(
            query=query,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            state_session_id=state_session_id,
        )
    else:
        logger.warning("no history, can't regenerate")
        return history


def revocery(query: dict, history: Sequence | None = None) -> tuple[str, Sequence]:
    """恢复到上一轮对话"""
    history = [] if history is None else list(history)
    if len(history) > 0:
        query_data, _ = history.pop(-1)
        if isinstance(query_data, str):
            query["text"] = query_data
        else:
            # 获取文件
            query["files"].append(query_data[0])
    return query, history


def combine_chatbot_and_query(
    query: dict,
    history: Sequence | None = None,
) -> Sequence:
    history = [] if history is None else list(history)
    query_text = query["text"]
    if query_text is None or (
        len(query_text.strip()) == 0 and len(query["files"]) == 0
    ):
        return history

    # 将图片放入历史记录中
    for x in query["files"]:
        print(f"file: {x}")
        history.append([(x,), None])
    return history + [[query_text, None]]


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
                        query = gr.MultimodalTextbox(
                            file_types=["image"],
                            file_count="multiple",  # 指的是一次上传几张,选择single也可以多次选择
                            placeholder="Enter 发送; Shift + Enter 换行 / Enter to send; Shift + Enter to wrap",
                            label="Prompt / 问题",
                            interactive=True,
                        )

                with gr.Row():
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
                        {"text": "你是谁", "files": []},
                        {
                            "text": "这张图片展示的什么内容?",
                            "files": ["images/0001.jpg"],
                        },
                        {
                            "text": "这2张图片展示的什么内容?",
                            "files": ["images/0001.jpg", "images/0002.jpg"],
                        },
                    ],
                    inputs=[query],
                    label="示例问题 / Example questions",
                )

            # 回车提交
            query.submit(
                multimodal_chat,
                inputs=[
                    query,
                    chatbot,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                    state_session_id,
                ],
                outputs=[chatbot],
            )

            # 清空query
            query.submit(
                lambda: gr.MultimodalTextbox(value={"text": "", "files": []}),
                inputs=[],
                outputs=[query],
            )

            # 拼接历史记录和问题
            query.submit(
                combine_chatbot_and_query,
                inputs=[query, chatbot],
                outputs=[chatbot],
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
                    state_session_id,
                ],
                outputs=[chatbot],
            )

            # 撤销
            undo.click(revocery, inputs=[query, chatbot], outputs=[query, chatbot])

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
