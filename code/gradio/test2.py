# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/gradio/turbomind_coupled.py
# https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/gradio/vl.py

# 导入必要的库
import gradio as gr
import numpy as np


def chat(
    query: str,
    history: list,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
    temperature: float = 0.8,
    regenerate: bool = False
) -> tuple[str, list]:
    """聊天"""
    # 重新生成时要把最后的query和response弹出,重用query
    if regenerate:
        # 有历史就重新生成,没有历史就返回空
        if len(history) > 0:
            query, _ = history.pop(-1)
        else:
            return "", history
    else:
        query = query.replace(' ', '')
        if query == None or len(query) < 1:
            return "", history

    print(
        {
            "max_new_tokens":  max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature}
    )

    history.append([query, str(np.random.randint(1, 100, 10))])
    return "", history


def regenerate(
    history: list,
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
    temperature: float = 0.8,
) -> tuple[str, list]:
    """重新生成最后一次对话的内容"""
    # 只返回history
    return chat("", history, max_new_tokens, top_p, top_k, temperature, regenerate=True)[1]


def revocery(history: list):
    """恢复到上一轮对话"""
    if len(history) > 0:
        history.pop(-1)
    return history


block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>InternLM2</center>
                """)
        # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建聊天框
            chatbot = gr.Chatbot(height=800, show_copy_button=True)

            with gr.Row():
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=2048,
                    value=1024,
                    step=1,
                    label='Maximum new tokens'
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
                temperature = gr.Slider(
                    minimum=0.01,
                    maximum=1.5,
                    value=0.8,
                    step=0.01,
                    label='Temperature'
                )

            with gr.Row():
                # 创建一个文本框组件，用于输入 prompt。
                query = gr.Textbox(label="Prompt/问题")
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

        # 回车提交
        query.submit(
            chat,
            inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature],
            outputs=[query, chatbot]
        )

        # 按钮提交
        submit.click(
            chat,
            inputs=[query, chatbot, max_new_tokens, top_p, top_k, temperature],
            outputs=[query, chatbot]
        )

        # 重新生成
        regen.click(
            regenerate,
            inputs=[chatbot, max_new_tokens, top_p, top_k, temperature],
            outputs=[chatbot]
        )

        # 撤销
        undo.click(
            revocery,
            inputs=[chatbot],
            outputs=[chatbot]
        )

    gr.Markdown("""提醒：<br>
    1. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)

# threads to consume the request
gr.close_all()

# 设置队列启动，队列最大长度为 100
demo.queue(max_size=100)

# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch(server_name="127.0.0.1", server_port=7860)
