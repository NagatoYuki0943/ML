# 导入必要的库
import gradio as gr
import numpy as np
from PIL import Image


labels = [
    "cat",
    "dog",
    "elephant",
    "giraffe",
    "horse",
    "sheep",
    "zebra",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "dining table",
    "potted plant",
    "sofa",
    "tv monitor",
]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def predict(
    image: Image.Image,
):
    logits = np.random.rand(20)
    logits = softmax(logits)
    confidence = {labels[i]: float(logits[i]) for i in range(len(labels))}
    return confidence


block = gr.Blocks()
with block as demo:
    with gr.Tab("图像分类"):
        gr.Markdown("图像分类演示")

        with gr.Row():
            image = gr.Image(
                sources=["upload", "webcam"],
                label="上传或拍照",
                image_mode="RGB",
                type="pil",
            )
            # 自动显示类别
            output_label = gr.Label(label="分类结果", num_top_classes=5)

        gr.Examples(
            examples=[
                ["images/doom之森01.jpg"],
                ["images/doom之森02.jpg"],
                ["images/doom之森03.jpg"],
                ["images/doom之森04.jpg"],
                ["images/doom之森05.jpg"],
            ],
            inputs=[image],
            label="示例图像",
        )

        submit = gr.Button("Predict", variant="primary", scale=0)

        # 按钮提交
        submit.click(predict, inputs=[image], outputs=[output_label])

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    max_threads=100,
)
