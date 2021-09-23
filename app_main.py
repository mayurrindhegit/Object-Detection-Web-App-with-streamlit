#streamlit run main_file.py
import io
import requests
import streamlit as st
import cv2 
import numpy as np  
import cvlib as cv
from PIL import Image
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt

def main():
    st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded")
    
    st.title("Object Detection\n\n\n\n")
    
    st.write('Upload any image of your choice ')
    input_image = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"]) # image upload widget
    
    if input_image:
        
        original_image = Image.open(input_image).convert("RGB")

        opencvimage = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        bbox , label , conf = cv.detect_common_objects(opencvimage , model = 'yolov3')
        output_img = draw_bbox(opencvimage , bbox , label , conf)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        col1, col2 = st.beta_columns(2)
        with col1:
            st.image(output_img)
        with col2:
            st.write("Objects Detected :{}".format(label))
        
main()


