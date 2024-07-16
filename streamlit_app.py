import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
from zipfile import ZipFile
import io

# Title
st.title('YOLO Image Annotation App')

# Layout
col1 = st.container()
col2, col3 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Select Image", type=["jpg", "jpeg", "png"])

with col2:
    st.header("Input")
    input_image = st.empty()

with col3:
    st.header("Output")
    output_image = st.empty()

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'annotated_img_path' not in st.session_state:
    st.session_state.annotated_img_path = None
if 'txt_data' not in st.session_state:
    st.session_state.txt_data = None

# Process the uploaded file
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

# Display the image
if st.session_state.annotated_img_path is not None:
    output_image.image(st.session_state.annotated_img_path, caption="Output Annotated Image", use_column_width=True)

if st.session_state.uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(st.session_state.uploaded_file)
    img_array = np.array(image)

    # Display the input image
    input_image.image(img_array, caption="Input Image", use_column_width=True)

    # Use columns for buttons to make them span the full width of their respective columns
    col4, col5 = st.columns([1, 1])
    
    with col4:
        process_button = st.button('Process', type='primary', use_container_width=True)

    if process_button and not st.session_state.processed:
        # Load the model
        GhostModel = YOLO('best.pt')

        # Save the uploaded image temporarily
        temp_img_path = "temp_uploaded_image.jpg"
        image.save(temp_img_path)

        # Perform inference
        results = GhostModel(temp_img_path, iou=0.5, line_width=3, save_txt=True)

        # Annotate the image
        annotated_img = results[0].plot()

        # Convert BGR image to RGB for displaying with matplotlib
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Save the annotated image
        st.session_state.annotated_img_path = "annotated_image.jpg"
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(st.session_state.annotated_img_path, annotated_img_rgb)

        # Find the latest prediction directory
        runs_dir = "runs/detect"
        latest_run_dir = max([os.path.join(runs_dir, d) for d in os.listdir(runs_dir)], key=os.path.getmtime)

        # Find the latest label file in the latest prediction directory
        label_dir = os.path.join(latest_run_dir, "labels")
        latest_label_file = max([os.path.join(label_dir, f) for f in os.listdir(label_dir)], key=os.path.getmtime)

        # Read the latest label file content
        with open(latest_label_file, 'r') as file:
            st.session_state.txt_data = file.read()

        # Display the image
        output_image.image(annotated_img, caption="Output Annotated Image", use_column_width=True)

        # Set the processed flag to True
        st.session_state.processed = True

    # Display the download button only if the image has been processed
    if st.session_state.processed:
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, 'w') as zip_file:
            # Add the annotated image to the zip file
            zip_file.write(st.session_state.annotated_img_path, arcname="annotated_image.jpg")
            # Add the results text file to the zip file
            zip_file.writestr("results.txt", st.session_state.txt_data)
        zip_buffer.seek(0)

        with col5:
            st.download_button(
                label="Download Results",
                use_container_width=True,
                data=zip_buffer,
                file_name="results.zip",
                mime="application/zip"
            )
