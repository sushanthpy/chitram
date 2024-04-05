import os
import requests
import time
import torch
import soundfile as sf
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from PIL import Image
from diffusers.utils import export_to_video
from transformers import AutoProcessor, MusicgenForConditionalGeneration


# Dynamic imports for heavy libraries
def dynamic_imports():
    global Groq, DiffusionPipeline, StableDiffusionInstructPix2PixPipeline, StableVideoDiffusionPipeline, EulerAncestralDiscreteScheduler, MusicGen, StableDiffusionXLImg2ImgPipeline
    from groq import Groq
    from diffusers import DiffusionPipeline, StableDiffusionInstructPix2PixPipeline, StableVideoDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionXLImg2ImgPipeline
    from audiocraft.models.musicgen import MusicGen

# Utilize GPU Acceleration and Optimizations
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner


class VideoMusicGenerator:
    """A class to generate a video with music based on a given prompt."""
    def __init__(self, project_name, prompt):
        self.project_name = project_name
        self.prompt = prompt
        self.project_folder = f"./{project_name}"
        self.cache_dir = './huggingface_models/'
        
        os.makedirs(self.project_folder, exist_ok=True)
        dynamic_imports()  # Perform dynamic imports 
        
        self.client = Groq(api_key=os.environ['GROQ_API_KEY'])
        self.image_path = os.path.join(self.project_folder, "image.png")
        self.refined_image_path = os.path.join(self.project_folder, "image_refine.png")
        self.video_path_no_music = os.path.join(self.project_folder, "final_output_video.mp4")
        self.extended_video_path = os.path.join(self.project_folder, "extended_video.mp4")
        self.edited_video_path = os.path.join(self.project_folder, "edited_video.mp4")
        self.music_path = os.path.join(self.project_folder, "generated_music.wav")
        self.final_video_path = os.path.join(self.project_folder, "final_video_with_music.mp4")
        self.n_steps = 40

    def enhance_prompt(self):
        api_key = os.environ["GENERATIVE_LANGUAGE_API_KEY"]
        content = f"Expand and enrich this passage with more details for generating image description, focusing solely on an enhanced narrative and short. {self.prompt}"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": content
                        }
                    ]
                }
            ]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)        
        if response.status_code == 200:
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"Enhanced prompt: {text}")
            return text
        else:
            print(f"Error: {response.status_code}")
            return None

    def enhance_prompt_groq(self):
        """Enhance the prompt using Groq API."""
        content = f"Expand and enrich this passage with more details, focusing solely on an enhanced narrative and short. {self.prompt}"
        chat_completion = self.client.chat.completions.create(
            messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model="mixtral-8x7b-32768",
            )
        groq_response = chat_completion.choices[0].message.content
        return groq_response    

    def recommend_instrument(self):
        """Recommend a musical instrument based on the prompt."""
        content = f"Given the following prompt: '{self.prompt}', recommend in  short a musical instrument that would be suitable for generating background music for a video based on this prompt. Provide only the name of the instrument."
        chat_completion = self.client.chat.completions.create(
            messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model="llama2-70b-4096",
            )
        instrument = chat_completion.choices[0].message.content
        print(f"Recommended instrument: {instrument}")
        return instrument

    def generate_image(self):
        """Generate an image based on the enhanced prompt."""
        base_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            force_download=True, resume_download=False).to("cuda")
        print("image generating prompt", self.enhance_prompt())
        image = base_pipe(
            prompt=self.enhance_prompt(),
            num_inference_steps=self.n_steps,
            height=576,
            width=1024,
        ).images[0]
        image.save(self.image_path)

    def refine_image(self):
        """Refine the generated image."""
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe = pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        img = Image.open(self.image_path).resize((1024, 576))
        image = pipe(self.enhance_prompt(), image=img, num_inference_steps=10, image_guidance_scale=1).images[0]
        image.save(self.refined_image_path)

    def extract_last_frame(self, video_path, image_path):
        """Extract the last frame from a video."""
        with VideoFileClip(video_path) as video:
            last_frame = video.get_frame(video.duration - 0.01)
        last_frame_image = Image.fromarray(last_frame)
        last_frame_image.save(image_path)
        return image_path
        

    def generate_video_from_image(self):
        """Generate a video from the refined image."""
        num_iterations = 3
        video_paths = []
        svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt-1-1", 
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        
        for iteration in range(num_iterations):
            image = Image.open(self.refined_image_path).resize((1024, 576))
            seed = int(time.time())
            torch.manual_seed(seed)
            frames = svd_pipe(image, decode_chunk_size=12, generator=torch.Generator(), motion_bucket_id=127).frames[0]
            video_path = os.path.join(self.project_folder, f"video_segment_{iteration}.mp4")
            export_to_video(frames, video_path, fps=5)
            video_paths.append(video_path)
            self.extract_last_frame(video_path, os.path.join(self.project_folder, "last_frame.png"))
        
        clips = [VideoFileClip(path) for path in video_paths]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(self.video_path_no_music)

    def generate_music(self):
        """Generate music based on the prompt."""
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        inputs = processor(
            text= f"{self.recommend_instrument} {self.enhance_prompt()}",
            padding=True,
            return_tensors="pt",
        )
        audio_values = model.generate(**inputs, max_new_tokens=512)
        sampling_rate = model.config.audio_encoder.sampling_rate
        sf.write(self.music_path, audio_values.squeeze().cpu().numpy(), samplerate=sampling_rate)
        
    def compose_final_video(self):
        """Compose the final video with music."""
        video_clip = VideoFileClip(self.video_path_no_music)
        audio_clip = AudioFileClip(self.music_path)

        min_duration = min(video_clip.duration, audio_clip.duration)

        video_clip = video_clip.subclip(0, min_duration)
        audio_clip = audio_clip.subclip(0, min_duration)

        final_audio = CompositeAudioClip([audio_clip])
        final_clip = video_clip.set_audio(final_audio)
        final_clip.write_videofile(self.final_video_path)

    def run(self):
        # Orchestrate the entire pipeline
        self.enhance_prompt()
        self.recommend_instrument()
        self.generate_image()
        self.refine_image()
        self.generate_video_from_image()
        self.generate_music()
        self.compose_final_video()

if __name__ == '__main__':
    project_name = "photo_riding_pig"
    #prompt = "create image of Imagine a breathtaking scene where a majestic waterfall cascades down a rugged cliff into a serene pool below, surrounded by lush green trees in the heart of a vibrant forest. The sun breaks through the canopy, casting shimmering light across the misty air. In the foreground, a colorful bird with outstretched wings glides gracefully over the water, embodying the spirit of freedom and the wild beauty of nature. The air is filled with the soothing sounds of water crashing into the pool and the distant calls of wildlife, creating a harmonious symphony of the wilderness."
    prompt = "A professional photograph of an astronaut riding a pig."
    generator = VideoMusicGenerator(project_name, prompt)
    generator.run()
