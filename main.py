from monsterapi import client
import cv2
import urllib
import numpy as np
import streamlit as st

api_key = "...."
# Get api key: https://monsterapi.ai/

def generate(prompt):
    monster_client = client(api_key)

    model = 'sdxl-base' 
    input_data = {
    'prompt': prompt,
    'negprompt': 'unreal, fake, meme, joke, disfigured, poor quality, bad, ugly',
    'samples': 2,
    'enhance': True,
    'optimize': True,
    'safe_filter': True,
    'steps': 50,
    'aspect_ratio': 'square',
    'guidance_scale': 7.5,
    'seed': 2414,
    }
    result = monster_client.generate(model, input_data)
    images = result['output']
    image = urllib.request.urlopen(images[0])
    arr = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img 


st.title('Object Detection for Images')
prompt = st.text_input(label="Prompt")
if st.button("Generate"):
    if prompt is not None:
        image = generate(prompt=prompt)
        st.image(image, caption="Generated Image ")
    else:
        st.write('Please write a prompt.')


