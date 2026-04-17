import streamlit as st
st.title('Object Detection + Tracking Demo')
st.write('This demo runs a YOLO detector and a simple IoU tracker over an uploaded image sequence or a sample toy dataset.')
if st.button('Create toy dataset (20 train, 5 val)'):
    import subprocess, sys
    subprocess.run([sys.executable, 'scripts/create_toy_dataset.py'])
    st.success('Toy dataset created at data/toy')
if st.button('Run short train (3 epochs)'):
    import subprocess, sys
    subprocess.run([sys.executable, 'src/train.py', '--data', 'data/toy/data.yaml', '--epochs', '3'])
    st.success('Training finished (check runs/detect/train)')
if st.button('Run detection+tracking on toy val'):
    import subprocess, sys
    subprocess.run([sys.executable, 'src/inference.py', '--weights', 'yolov8n.pt', '--source', 'data/toy/images/val'])
    st.success('Detection+tracking complete. Visualizations in runs/detect/track')

st.write('To run locally: pip install -r requirements.txt and ensure ultralytics is installed. The app invokes local scripts and displays status only.')
