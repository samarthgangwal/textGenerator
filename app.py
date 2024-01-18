import streamlit as slit
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
#from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
import torch

slit.set_page_config(page_title="Text Generator",page_icon='📜',layout='wide',initial_sidebar_state='collapsed')

slit.header("Text Generator 📜")

textInput = slit.text_input("Enter the text topic:")

def LLAmaresponse(textInput,numWords,textStyle):

    ###LLAma2 Model###
    llm = CTransformers(model='Models\llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens':256,
                                'temperature':0.01})  

    ###Template###

    template="Generate text for {textStyle} for a topic {textInput}. Having {numWords} words."

    prompt = PromptTemplate(input_variables=["textStyle","textInput","numWords"],
                            template=template)
    
    '''LLAma2 Response Generation'''
    res = llm(prompt.format(textStyle=textStyle,textInput=textInput,numWords=numWords))
    print(res)
    return res

c1,c2 = slit.columns([5,5])

with c1:
    numWords = slit.text_input("Number of Words")

with c2:
    textStyle = slit.selectbox('Generating text for',('General','Research'),index=0)

sub = slit.button("Generate")

###Response###

if sub:
    slit.write(LLAmaresponse(textInput,numWords,textStyle))


