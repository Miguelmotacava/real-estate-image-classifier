import streamlit as st
import requests

st.title("ML Image Classifier")
st.write("Upload an image to get a prediction from our API.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image locally
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        # Prepare the file to send to the API
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        with st.spinner('Waiting for API response...'):
            try:
                # Call the FastAPI backend
                response = requests.post("http://localhost:8000/predict", files=files)
                result = response.json()
                
                # Display results
                st.success(f"Prediction: {result['label']}")
                st.metric("Confidence", f"{result['confidence']*100}%")
                
            except Exception as e:
                st.error(f"Error connecting to API: {e}")