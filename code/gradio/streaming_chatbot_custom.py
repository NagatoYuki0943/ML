# https://www.gradio.app/playground

import time
import gradio as gr


def slow_echo(
    query: str,
    history: list | None,  # [['What is the capital of France?', 'The capital of France is Paris.'], ['Thanks', 'You are Welcome']]
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
    temperature: float = 0.8,
):
    query = query.replace(' ', '')
    if query == None or len(query) < 1:
        for i in range(0):
            yield None

    print(
        {
            "max_new_tokens":  max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature
        }
    )

    for i in range(len(query)):
        time.sleep(0.1)
        yield "You typed: " + query[: i + 1]


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


demo = gr.ChatInterface(
    slow_echo,
    additional_inputs=[max_new_tokens, top_p, top_k, temperature],
    additional_inputs_accordion=gr.Accordion(label="Parameters", open=False)
).queue()


if __name__ == "__main__":
    demo.launch()
