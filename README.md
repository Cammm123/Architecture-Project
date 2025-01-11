# Architecture-Project
Generate an Image of any Given Prompt (Created for Architectural purposes but can be used for a wide variety of image generation needs)

How this pipeline works:
1) Insert a text prompt of what you want to see (e.g. A kitchen with cabinets made from oak wood)
2) This prompt gets passed into an LLM (I used Meta-Llama3.1-8B-Instruct), where several more details are added to the prompt to aid in image gernation
3) The new prompt gets passed into Stabble Diffusion (text to image model that is pre-trained)
4) The image generated from step 3 gets passed into an image upscaler pipline to make the image more realistic (this is important since the project was made for Architectural Purposes)
5) Finally, you can make slight tweeks to the image by editing your prompt and letting the pipeline work through again.

Notes: 
- This code was written in Anaconda's Spyder, and I think it might only work in that IDE (from my experience it doesn't work well in VSCode)
- You need to install several dependencies including Nvidia Cuda and a corresponding version for PyTorch (I reccomend finding which PyTorch models go with which Cuda Version since PyTorch is more restrictive)