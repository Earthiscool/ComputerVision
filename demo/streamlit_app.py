import streamlit as st
st.title('Object Detection + Tracking Demo (Tiny)')
st.write('This demo uses a lightweight YOLO model if installed. For local testing, run: python scripts/create_toy_dataset.py && python src/train.py')
file = st.file_uploader('Upload an image', type=['jpg','png','jpeg'])
if file is not None:
    from PIL import Image
    img = Image.open(file)
    st.image(img, caption='Uploaded image', use_column_width=True)
    st.write('To run detection locally, install ultralytics and run src/inference.py')
