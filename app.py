"""Create a Gradio interface for the VideoMusicGenerator class.

This script creates a Gradio interface for the VideoMusicGenerator class in chitram.py.
The interface allows users to input a project name and a prompt, and generates a video with music based on the prompt.
"""

import gradio as gr
from chitram import VideoMusicGenerator 


def run_generator(project_name, prompt):
    """Run the VideoMusicGenerator class."""
    generator = VideoMusicGenerator(project_name, prompt)
    generator.run() 
    return generator.final_video_path

def create_gradio_interface():
    """Create a Gradio interface for the VideoMusicGenerator class."""
    with gr.Blocks() as demo:
        gr.Markdown("## Chitram")
        gr.Markdown("Chitram is a tool that generates videos with music based on a given prompt.")
        with gr.Row():
            project_name = gr.Textbox(label="Project Name")
            prompt = gr.Textbox(label="Prompt", lines=4)
        
        generate_btn = gr.Button("Generate")
        output_video = gr.Video(label="Generated Video")

        generate_btn.click(
            fn=run_generator,
            inputs=[project_name, prompt],
            outputs=[output_video]
        )

    demo.launch()

if __name__ == "__main__":
    create_gradio_interface()
