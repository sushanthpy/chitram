# Chitram - VideoMusicGenerator

## Overview
VideoMusicGenerator is an innovative tool designed to create captivating videos accompanied by custom-generated music. By leveraging advanced AI models, it transforms textual prompts into a seamless integration of visually appealing imagery and thematic music, providing a unique multimedia experience. Ideal for content creators, marketers, and anyone looking to bring their creative visions to life, VideoMusicGenerator offers an intuitive interface for generating professional-grade video content with ease.

## Features
- **Dynamic Image and Video Creation:** Generate high-quality images and videos from textual prompts using state-of-the-art AI models.
- **Custom Music Composition:** Automatically compose music that fits the theme of the generated video content.
- **Flexible Model Selection:** Choose from various models for image generation, video creation, and music composition to suit your specific needs.
- **Gradio Interface:** An easy-to-use web interface powered by Gradio, allowing for dynamic configuration and real-time generation.

## Installation
Ensure you have Python 3.10+ installed on your system. Then, follow these steps to set up the VideoMusicGenerator:

1. **Clone the Repository**
    ```
    git clone https://github.com/sushanthpy/chitram.git
    cd chitram
    ```
2. **Install Dependencies**
    Install all the required Python packages using pip:
    ```
    pip install -r requirements.txt
    ```
3. **Set Up Environment Variables**
    Make sure to set the necessary environment variables, such as the API key for Groq:
    ```
    export GROQ_API_KEY='your_api_key_here'
    export GENERATIVE_LANGUAGE_API_KEY='google pro key'
    ```
4. **Run the Application**
    Start the Gradio interface by running the following command:
    ```
    python app.py
    ```

## Usage
To run the chitram - VideoMusicGenerator with the Gradio interface:

This command will start a local web server and print a URL to access the Gradio interface. Open this URL in a web browser to interact with the application.

## Configuration Options
- **Project Name:** Set a unique name for each project to organize your generated content.
- **Prompt:** Enter a descriptive text prompt to guide the content generation.
- **Image Model:** Select the model for image generation.
- **Number of Inference Steps:** Adjust the detail level in the generated images.
- **Video Model:** Choose the model for video creation.
- **Instrument:** Select a musical instrument for the background music composition.

## Contributing
Contributions to VideoMusicGenerator are welcome! Please refer to the CONTRIBUTING.md file for guidelines on how to contribute to this project.

Inspire by paper: https://huggingface.co/papers/2403.13248

## Things to do
- [x] Add more models for image generation, video creation, and music composition.
- [ ] Implement additional features for customization and control over the generated content.
- [ ] Optimize the performance and efficiency of the AI models.
- [ ] Enhance the user interface and user experience of the application.
- [ ] Autogen Agent Framework for multiagent system.
- [ ] Dockerize the application for easy deployment and scalability.

## Sample Output
![Sample Output]( cowboy_rider/final_video_with_music.mp4)

## License
Chitram - VideoMusicGenerator is licensed under the Apache License. See the LICENSE file for more details.

