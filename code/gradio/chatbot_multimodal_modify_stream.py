# 导入必要的库
import gradio as gr
import numpy as np
from typing import Generator, Sequence, Any
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


enable_btn = gr.update(interactive=True)
disable_btn = gr.update(interactive=False)
btn = dict[str, Any]


def multimodal_chat(
    query: dict,
    history: Sequence | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    state_session_id: int = 0,
) -> Generator[tuple[Sequence, btn, btn, btn], None, None]:
    logger.info(f"{state_session_id = }")

    history = [] if history is None else list(history)
    logger.debug(f"old history: {history}")

    logger.info(
        {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    )

    logger.info(f"query: {query}")
    query_text = query.get("text", "").strip()
    qyery_files = query.get("files", [])
    logger.info(f"query_text: {query_text}")
    logger.info(f"query_files: {qyery_files}")
    if len(query_text) == 0 and len(qyery_files) == 0:
        logger.warning("query is None, return history")
        yield history, enable_btn, enable_btn, enable_btn
        return

    # 将图片放入历史记录中
    for file in qyery_files:
        history += [
            {
                "role": "user",
                "content": {
                    "type": "image_url",
                    # path for gradio
                    "path": file,
                    # useless for gradio
                    "image_url": {
                        "url": file,
                    },
                },
            }
        ]
    if query_text is not None and len(query_text) > 0:
        history += [
            {
                "role": "user",
                "content": query_text,
            }
        ]

    yield history, disable_btn, disable_btn, disable_btn

    time.sleep(1)
    number: np.ndarray = np.random.randint(1, 100, 20)
    logger.info(f"number: {number}")
    logger.info(f"new history: {history}")

    _history = history
    for i in range(len(number)):
        time.sleep(0.1)
        logger.info(f"number[{i}] = {number[i]}")
        _history = history + [
            {
                "role": "assistant",
                "content": str(number[: i + 1]),
            },
        ]
        yield (
            _history,
            disable_btn,
            disable_btn,
            disable_btn,
        )

    logger.info(f"new history: {_history}")
    yield _history, enable_btn, enable_btn, enable_btn


def regenerate(
    history: Sequence | None = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 40,
    state_session_id: int = 0,
) -> Generator[tuple[Sequence, btn, btn, btn], None, None]:
    history = [] if history is None else list(history)
    logger.debug(f"old history: {history}")

    content = ""
    for message in history[::-1]:
        # 无论如何都删除后面的值
        history.pop()
        if message["role"] == "user":
            content = message["content"]
            break
    logger.debug(f"content: {content}")
    query = {}
    if isinstance(content, str):
        query["text"] = content.strip()
    elif isinstance(content, tuple) and len(content) > 0:
        query["files"] = [content[0]]

    if len(query.get("text", "")) > 0 or len(query.get("files", [])) > 0:
        yield from multimodal_chat(
            query,
            history,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            state_session_id,
        )
    else:
        logger.warning("no history, can't regenerate")
        yield history, enable_btn, enable_btn, enable_btn


def undo(query: dict, history: Sequence | None = None) -> tuple[str, Sequence]:
    """恢复到上一轮对话"""
    history = [] if history is None else list(history)

    query = ""
    for message in history[::-1]:
        # 无论如何都删除后面的值
        history.pop()
        if message["role"] == "user":
            content = message["content"]
            if isinstance(content, str):
                query = content
            break

    return query, history


def main():
    block = gr.Blocks()
    with block as demo:
        state_session_id = gr.State(0)

        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>🦞 Lobster</center></h1>""")
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    # 创建聊天框
                    chatbot = gr.Chatbot(
                        type="messages",
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
                    retry_btn = gr.Button("🔄 Retry", variant="secondary")
                    undo_btn = gr.Button("↩️ Undo", variant="secondary")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear_btn = gr.ClearButton(
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

            # 回车提交(无法禁止按钮)
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
                outputs=[chatbot, retry_btn, undo_btn, clear_btn],
            )

            # 清空query
            query.submit(
                lambda: gr.MultimodalTextbox(value={"text": "", "files": []}),
                inputs=[],
                outputs=[query],
            )

            # 重新生成
            retry_btn.click(
                regenerate,
                inputs=[
                    chatbot,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                    state_session_id,
                ],
                outputs=[chatbot, retry_btn, undo_btn, clear_btn],
            )

            # 撤销
            undo_btn.click(undo, inputs=[query, chatbot], outputs=[query, chatbot])

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
