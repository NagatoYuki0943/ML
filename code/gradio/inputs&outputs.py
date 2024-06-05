import gradio as gr


input_list: list = [
    gr.Textbox(label="Textbox", lines=3, max_lines=7, placeholder="Enter text..."),
    gr.TextArea(label="TextArea", lines=3, max_lines=7, placeholder="Enter text..."),
    gr.Number(label="Number", value=0, scale=15, interactive=True),
    gr.Slider(minimum=0, maximum=100, value=50, label="Slider"),
    gr.Checkbox(value=True, label="Checkbox"),
    gr.CheckboxGroup(choices=[1, 2, 3, 4], value=[1, 2], label="Checkbox Group"),
    gr.Dropdown(choices=[1, 2, 3, 4], value=1, type="value", label="Dropdown"),
    gr.Radio(choices=[1, 2, 3, 4], value=1, type="value", label="Radio"),
    gr.ColorPicker(label="Color Picker"),
    gr.Image(sources=["upload", "webcam", "clipboard"], type="pil", label="Image"),
    gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio File"),
    gr.Video(sources=["upload", "webcam"], label="Video"),
    gr.File(label="File", type="binary"),
    gr.UploadButton(label="upload a file", file_types=["image", "audio", "video", "text"]),
    gr.DataFrame(headers=["id", "name", "age"], label="Dataframe"),
]

output_list: list = [
    gr.Textbox(label="Textbox output"),
    gr.TextArea(label="TextArea output"),
    gr.Number(label="Number output"),
    gr.Slider(label="Slider output"),
    gr.Checkbox(label="Checkbox output"),
    gr.CheckboxGroup(label="Checkbox Group output"),
    gr.Dropdown(label="Dropdown output"),
    gr.Radio(label="Radio output"),
    gr.ColorPicker(label="Color Picker output"),
    gr.Image(label="Image output"),
    gr.Audio(label="Audio File output"),
    gr.Video(label="Video output"),
    gr.File(label="File output"),
    gr.File(label="UploadButton output"),
    gr.DataFrame(label="Dataframe output"),
]


def input_and_output(*input_data) -> tuple:
    return input_data


interface = gr.Interface(
    fn=input_and_output,
    inputs=input_list,
    outputs=output_list,
    title="Input and Output",
    description="Input and Output Demo",
    live=True
)
interface.launch()
