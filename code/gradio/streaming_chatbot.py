"""
https://www.gradio.app/docs/gradio/chatinterface
https://www.gradio.app/guides/creating-a-chatbot-fast

"""

import gradio as gr


def yes_man(message: str, history: list):
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"


gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(type="messages", height=300),
    textbox=gr.Textbox(
        placeholder="Ask me a yes or no question", container=False, scale=7
    ),
    title="Yes Man",
    description="Ask Yes Man any question",
    theme="soft",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
    submit_btn="💬 Chat",
    stop_btn="🚫 Stop",
).launch()
