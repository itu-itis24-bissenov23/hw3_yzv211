import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

### DO NOT EDIT IMPORTS ABOVE THIS LINE ###

# -----------------------------------------------------------------------------
# STEP 1: MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_caption_model():
    # TODO: Load the Processor (Handles image resizing & tokenization)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # TODO: Load the Model 
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# TODO
processor, model = load_caption_model()

# -----------------------------------------------------------------------------
# STEP 2: UI LAYOUT
# -----------------------------------------------------------------------------
# TODO: set title and sidebar
st.title("The Image Captioner")
st.sidebar.header("Parameter Settings")

# TODO: set sidebar sliders

# 1. Temperature (min_value=0.1, max_value=1.5, value=0.1, step=0.1)

temperature = st.sidebar.slider(
    label="Temperature",
    min_value=0.1,
    max_value=1.5,
    value=1.0,
    step=0.1,
    help="Choosing higher values provides higher randomness")

# 2. Max Length (min_value=5, max_value=30, value=20, step=1)

max_length = st.sidebar.slider(
    label="Max Length (Tokens)",
    min_value=5,
    max_value=30,
    value=20,
    step=1)

# 3. Min Length (min_value=3, max_value=20, value=5, step=1)

min_length = st.sidebar.slider(
    label="Min Length (Tokens)",
    min_value=3,
    max_value=20,
    value=5,
    step=1)

# 4. Number of Variations (min_value=1, max_value=5, value=1, step=1)

num_captions = st.sidebar.slider(
    label="Number of Variations",
    min_value=1,
    max_value=5,
    value=1,
    step=1)

st.sidebar.markdown("---")
st.sidebar.write("Developed by: **[Serikbol Bissenov]**")
st.sidebar.write("Student ID: **[150230922]**")

# -----------------------------------------------------------------------------
# STEP 3: MAIN PIPELINE
# -----------------------------------------------------------------------------
# TODO: Introduction text under title

st.write("Upload an image for the AI")

# TODO: File uploader for images, accept jpg, jpeg, png

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # TODO: If an image is uploaded, display it
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # TODO: Create a input box for starting text prompt
    start_text = st.text_input("Write your sentence: ", placeholder="e.g., A photo of a")

    # # TODO: Create a button called 'Generate Caption' to trigger the AI
    if st.button("Generate Caption"):
        with st.spinner("Generating caption"):
        # You may add a loading spinner here while the model is generating captions
            
            # TODO: Display a subheader 'Caption(s):'
            st.subheader("Caption(s):")

            # TODO: PRE-PROCESSING (Convert Image & Text to Tensors) ---
            # If user provided start_text, we pass it as 'text'. 
            # Otherwise we just pass the image.
            if start_text.strip():
                inputs = processor(image, text=start_text, return_tensors="pt")
            else:
                inputs = processor(image, return_tensors="pt")
            
            # TODO: Create a loop that runs 'num_captions' times
            for i in range(num_captions):
                # TODO: Pass the inputs and parameters to generate caption(s).
                # Some parameters are already given.
                # It should also contain the parameter values given by the user via the sliders.
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    min_length=min_length,
                    max_length=max_length,
                    top_k=50
                )

                # TODO: POST-PROCESSING (Decode Tensors back to Text)
                # The model gives us numbers, we need to decode them to words.
                # You may omit special tokens.
                caption_text = processor.decode(out[0], skip_special_tokens=True)
            
                # TODO: Display the result(s) depending on number of variations.
                st.success(f"{i + 1}. {caption_text}")