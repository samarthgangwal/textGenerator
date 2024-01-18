import streamlit as slit
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
#from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
import cv2
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cpu")
generator = torch.Generator("cpu").manual_seed(31) 
image = pipeline('Horse', generator=generator).images[0]

img = cv2.imread(image)
cv2.imshow('Sample',img)
cv2.waitKey(0) 
cv2.destroyAllWindows() 
