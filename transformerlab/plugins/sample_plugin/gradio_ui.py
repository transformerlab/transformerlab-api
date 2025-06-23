import gradio as gr
from datetime import datetime


def process_text(input_text):
    """
    Simple text processing function.
    Replace this with your actual processing logic.
    """
    if not input_text.strip():
        return "Please enter some text!"

    # Simple example: return the text with some processing
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"You entered: {input_text}\nProcessed at: {timestamp}"


# Create a simple Gradio interface
with gr.Blocks(title="Simple Plugin", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Simple Text Processor")

    input_textbox = gr.Textbox(label="Enter text", placeholder="Type something here...", lines=3)

    process_button = gr.Button("Process Text", variant="primary")

    output_textbox = gr.Textbox(label="Output", lines=5, interactive=False)

    # Connect the button to the function
    process_button.click(
        fn=process_text,
        inputs=input_textbox,
        outputs=output_textbox,
        queue=False,  # Disable queueing for this specific interaction
    )

# Required for the FastAPI server
app = demo
