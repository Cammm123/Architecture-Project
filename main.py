# Import necessary libraries
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
from transformers import LlamaForCausalLM, AutoTokenizer, pipeline
from PIL import Image

# Set up GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Check PyTorch CUDA setup
print("PyTorch CUDA available:", torch.cuda.is_available())
if device == "cuda":
    print("PyTorch CUDA version:", torch.version.cuda)
    print("CUDA Device:", torch.cuda.get_device_name(0))

#%% Stable Diffusion - Initial Image Generation
"""
Stable Diffusion Creates Image Based on Descriptive Text
"""

# Initialize the Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Prompt for initial image generation
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.show()

#%% LLaMA Model - Text Generation
"""
Llama Model Creates Detailed Descriptive Text
"""

# Set the path to your local LLaMA model
model_path = #insert path to your model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16).to(device)

# Initialize the pipeline for text generation
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    torch_dtype=torch.float16
)

# Enhanced input prompt for more detailed description
input_text = """Generate a detailed description of a luxurious modern kitchen. Include specific details about:
- Cabinet materials, color, and style
- Countertop materials and finish
- Lighting fixtures
- Appliances
- Layout and space
- Flooring
- Any unique architectural features
Make the description vivid and specific, focusing on visual elements."""

# Generate the description with improved parameters
sequences = generator(
    input_text,
    do_sample=True,
    top_k=50,          # Increased from 10 to allow more creative variety
    top_p=0.9,         # Added to help with text coherence
    temperature=0.7,   # Added to balance creativity and coherence
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,    # Increased for longer description
    min_length=100,    # Added to ensure detailed description
    repetition_penalty=1.2  # Added to prevent repetitive text
)

# Extract and process the generated description
generated_description = sequences[0]['generated_text']

# Add key phrases that help Stable Diffusion
enhanced_description = f"""professional architectural visualization, {generated_description}, 
photorealistic, highly detailed, architectural photography, interior design magazine quality, 
8k uhd, ray tracing, volumetric lighting"""

print(f"Enhanced Description:\n{enhanced_description}")

#%% Stable Diffusion - Generate Image Based on Description
# Create output directory
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Update the image generation parameters for better quality
image = pipe(
    enhanced_description,
    num_inference_steps=50,  # Increased for better quality
    guidance_scale=7.5,      # Controls how closely to follow the prompt
    negative_prompt="blur, distortion, disfigured, bad architecture, poor lighting, bad proportions, watermark"
).images[0]

# Save the initial generated image
initial_image_path = os.path.join(output_dir, "initial_kitchen.png")
image.show()
image.save(initial_image_path)

#%% Stable Diffusion Upscaler - Enhance Image Realism
"""
Upscale and Enhance the Image for Realism
"""

# Initialize the upscaler pipeline
upscaler = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=torch.float16
).to(device)

# Load the generated image
low_res_img = Image.open(initial_image_path).convert("RGB")

# Upscale the image with enhanced details
upscaled_image = upscaler(
    prompt="professional photo of a modern kitchen, highly detailed, realistic, architectural visualization",
    image=low_res_img,
    noise_level=20,
    num_inference_steps=30
).images[0]

# Save the upscaled image
refined_output_path = os.path.join(output_dir, "refined_kitchen.png")
upscaled_image.save(refined_output_path)

print(f"Enhanced image saved at: {refined_output_path}")
image.show()
#%% Modifying Generated Image
def generate_modified_image(base_description, modification, pipe, output_dir, upscaler):
    """
    Generate a new image based on the original description with specific modifications
    """
    # Combine base description with modification
    modified_description = f"""professional architectural visualization, {base_description},
    WITH MODIFICATIONS: {modification},
    photorealistic, highly detailed, architectural photography, interior design magazine quality,
    8k uhd, ray tracing, volumetric lighting"""
    
    print(f"Generating image with modification: {modification}")
    print(f"Full prompt:\n{modified_description}")
    
    # Generate new image
    image = pipe(
        modified_description,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt="blur, distortion, disfigured, bad architecture, poor lighting, bad proportions, watermark"
    ).images[0]
    
    # Save the initial modified image
    initial_image_path = os.path.join(output_dir, f"initial_kitchen_mod_{len(os.listdir(output_dir))}.png")
    image.save(initial_image_path)
    image.show()
    
    # Upscale the image
    low_res_img = Image.open(initial_image_path).convert("RGB")
    upscaled_image = upscaler(
        prompt="professional photo of a modern kitchen, highly detailed, realistic, architectural visualization",
        image=low_res_img,
        noise_level=20,
        num_inference_steps=30
    ).images[0]
    
    # Save the upscaled modified image
    refined_output_path = os.path.join(output_dir, f"refined_kitchen_mod_{len(os.listdir(output_dir))}.png")
    upscaled_image.save(refined_output_path)
    upscaled_image.show()
    
    return upscaled_image

# Store the original description without the enhancement keywords
base_description = generated_description

while True:
    modification = input("Enter desired modification (or 'quit' to exit): ")
    if modification.lower() == 'quit':
        break
    modified_image = generate_modified_image(base_description, modification, pipe, output_dir, upscaler)
    print(f"Generated variation with: {modification}")