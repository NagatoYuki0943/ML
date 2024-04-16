# https://www.gradio.app/playground

import time
import gradio as gr


def slow_echo(query, history):
    query = query.replace(' ', '')
    if query == None or len(query) < 1:
        for i in range(1):
            yield None

    print("message: ", query)
    print("history: ", history)

    for i in range(len(query)):
        time.sleep(0.1)
        yield "You typed: " + query[: i + 1]


demo = gr.ChatInterface(slow_echo).queue()


if __name__ == "__main__":
    demo.launch()
