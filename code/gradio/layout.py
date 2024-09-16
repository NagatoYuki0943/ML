import gradio as gr


with gr.Blocks(css="style.css") as demo:
    with gr.Tab(label="txt2img"):
        with gr.Row():
            with gr.Column(scale=15):
                txt1 = gr.Textbox(lines=2, label="txt1", placeholder="txt1")
                txt2 = gr.Textbox(lines=2, label="txt2")

            # 设置最小宽度
            with gr.Column(scale=1, min_width=5):
                botton1 = gr.Button(value="1", elem_classes="btn")
                botton2 = gr.Button(value="2", elem_classes="btn")
                botton3 = gr.Button(value="3", elem_classes="btn")
                botton4 = gr.Button(value="4", elem_classes="btn")

            with gr.Column(scale=5):
                gen_botton = gr.Button(value="Generation", variant="primary", scale=1)
                with gr.Row():
                    dropdown1 = gr.Dropdown(
                        choices=[1, 2, 3, 4],
                        value=1,
                        label="style1",
                        type="value",
                        interactive=True,
                    )
                    dropdown2 = gr.Dropdown(
                        choices=[1, 2, 3, 4],
                        value=2,
                        label="style2",
                        type="value",
                        interactive=True,
                    )

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    dropdown3 = gr.Dropdown(
                        choices=[1, 2, 3, 4],
                        value=3,
                        label="Sampling Method",
                        type="value",
                        interactive=True,
                    )
                    slider1 = gr.Slider(
                        label="Sampling Steps",
                        minimum=1,
                        maximum=100,
                        value=25,
                        step=1,
                        interactive=True,
                    )

                checkboxgroup = gr.CheckboxGroup(
                    choices=["Resotre Faces", "Tiling", "Hires.fix"],
                    label="CheckboxGroup",
                    value=["Resotre Faces"],
                    type="value",
                    interactive=True,
                )

                with gr.Accordion("Open for More!", open=True):
                    with gr.Row():
                        slider2 = gr.Slider(
                            label="Width",
                            minimum=100,
                            maximum=1000,
                            value=500,
                            step=1,
                            interactive=True,
                        )
                        slider3 = gr.Slider(
                            label="Batch Count",
                            minimum=1,
                            maximum=100,
                            value=1,
                            step=1,
                            interactive=True,
                        )

                    with gr.Row():
                        slider4 = gr.Slider(
                            label="Height",
                            minimum=100,
                            maximum=1000,
                            value=500,
                            step=1,
                            interactive=True,
                        )
                        slider5 = gr.Slider(
                            label="Batch Count",
                            minimum=1,
                            maximum=100,
                            value=1,
                            step=1,
                            interactive=True,
                        )

                slider6 = gr.Slider(
                    label="CFG Scale",
                    minimum=1,
                    maximum=100,
                    value=70,
                    step=1,
                    interactive=True,
                )

                with gr.Row():
                    number1 = gr.Number(
                        label="Seed", value=0, scale=15, interactive=True
                    )
                    button5 = gr.Button(
                        value="Random", variant="secondary", scale=1, min_width=10
                    )
                    button6 = gr.Button(
                        value="Reset", variant="secondary", scale=1, min_width=10
                    )
                    checkbox1 = gr.Checkbox(
                        label="Extra",
                        value=True,
                        scale=4,
                        min_width=10,
                        interactive=True,
                    )

                dropdown3 = gr.Dropdown(
                    choices=[1, 2, 3, 4],
                    value=2,
                    label="Script",
                    type="value",
                    interactive=True,
                )

            with gr.Column():
                image1 = gr.Gallery(
                    value=[
                        "images/doom之森01.jpg",
                        "images/doom之森02.jpg",
                        "images/doom之森03.jpg",
                        "images/doom之森04.jpg",
                        "images/doom之森05.jpg",
                    ],
                    columns=2,  # 默认显示的宽高
                    rows=2,
                    label="image1",
                )

                # 组内的组件没有间距
                with gr.Group():
                    with gr.Row():
                        button7 = gr.Button(
                            value="C", variant="secondary", min_width=10
                        )
                        button8 = gr.Button(
                            value="Save", variant="secondary", min_width=10
                        )
                        button9 = gr.Button(
                            value="Zip", variant="secondary", min_width=10
                        )
                        button10 = gr.Button(
                            value="Send to img2img", variant="secondary", min_width=10
                        )
                        button11 = gr.Button(
                            value="Send to inpaint", variant="secondary", min_width=10
                        )
                        button12 = gr.Button(
                            value="Send to extras", variant="secondary", min_width=10
                        )

                txt3 = gr.Textbox(lines=4, label="txt3", placeholder="txt3")

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    max_threads=100,
)
