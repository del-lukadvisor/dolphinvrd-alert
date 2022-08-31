from distutils.log import error
import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import streamlit_modal as modal
import streamlit.components.v1 as components

##########
##### Set up sidebar.
##########

# Add in location to select image.

## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)

#image = Image.open('example/000001/000000.png')
#st.sidebar.image(image,
#                 use_column_width=True)

#image = Image.open('example/000001/000000.png')
#st.sidebar.image(image,
#                 use_column_width=True)

##########
##### Set up main app.
##########

## Title.
st.write('# Dolphin Detection')

def custom_modal():
    open_modal = st.button("Warning 🚨")
    if open_modal:
         modal.open()
    if modal.is_open():
        with modal.container():
            html_string = '''
            <div>
            <h1>Warning 🚨</h1>
            <p>Abnormalities detected on inferenced image/video</p>
            </div>

            <script language="javascript">
            document.querySelector("div").style.padding = "10px";
            document.querySelector("h1").style.color = "red";
            document.querySelector("p").style.color = "black";
            document.querySelector("p").style.fontSize = "18px";
            </script>
            '''
            components.html(html_string)

st.write('#### Select image or video to upload.')
uploaded_file = st.file_uploader('',
                                         type=['png', 'jpg', 'jpeg', 'mp4', 'avi'],
                                         accept_multiple_files=False)


## Pull in default image or user-selected image.
#uploaded_file = 'example/000000.gif'   # 1/000000.png'
#uploaded_file = 'example/000001/000000.png'   # 1/000000.png'


if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    custom_modal()
    # User-selected image.
    image = Image.open(uploaded_file)

## Subtitle.
st.write('### Inferenced Image/Video')

# Convert to JPEG Buffer.
buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')

# Base 64 encode.
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')

## Construct the URL to retrieve image.


#image = Image.open(BytesIO(r.content))

# Convert to JPEG Buffer.
buffered = io.BytesIO()
#image.save(buffered, quality=90, format='JPEG')

# Display image.
#st.image(image, use_column_width=True)

file_ = open("example/000000.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

if uploaded_file:
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="alt gif">',
        unsafe_allow_html=True
    )

    ## Construct the URL to retrieve JSON.
    upload_url = ''.join([
        'https://infer.roboflow.com/rf-bccd-bkpj9--1',
        '?access_token=vbIBKNgIXqAQ'
    ])

    ## POST to the API.
    r = requests.post(upload_url,
                      data=img_str,
                      headers={
        'Content-Type': 'application/x-www-form-urlencoded'
    })

    ## Save the JSON.
    output_dict = r.json()

    ## Generate list of confidences.
    #onfidences = [box['confidence'] for box in output_dict['predictions']]
    confidences = [0.7, 0.6, 0.5]

    ## Summary statistics section in main app.
    st.write('### Summary Statistics')
    st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
    st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

    ## Histogram in main app.
    st.write('### Histogram of Confidence Levels')
    fig, ax = plt.subplots()
    ax.hist(confidences, bins=10, range=(0.0,1.0))
    st.pyplot(fig)


    form = st.form(key='my-form')
    submit = form.form_submit_button('Export JSON File')

    if submit:
        st.write(f'Saved Json file')