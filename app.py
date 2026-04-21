import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import urllib.request
import os

# මෙතනට ඔයාගේ Release ලින්ක් එක දාන්න (උද්ධෘත ලකුණු ඇතුළේ)
MODEL_URL = "මෙතන_ඔයාගේ_ලින්ක්_එක_දාන්න"

@st.cache_resource
def load_model():
    model_path = 'crop_disease_model.h5'
    
    # සර්වර් එකේ ෆයිල් එක නැත්නම් අර ලින්ක් එකෙන් බාගන්නවා
    if not os.path.exists(model_path):
        st.info("AI මොළය භාගත කරමින් පවතී... (මෙයට විනාඩියක් පමණ ගත විය හැක)")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

class_names = [
    'Pepper bell (මාළු මිරිස්) - Bacterial spot',
    'Pepper bell (මාළු මිරිස්) - Healthy (නිරෝගී)',
    'Potato (අර්තාපල්) - Early blight (පෙරදිග අංගමාරය)',
    'Potato (අර්තාපල්) - Late blight (පසුදිග අංගමාරය)',
    'Potato (අර්තාපල්) - Healthy (නිරෝගී)',
    'Tomato (තක්කාලි) - Bacterial spot',
    'Tomato (තක්කාලි) - Early blight',
    'Tomato (තක්කාලි) - Late blight',
    'Tomato (තක්කාලි) - Leaf Mold',
    'Tomato (තක්කාලි) - Septoria leaf spot',
    'Tomato (තක්කාලි) - Spider mites',
    'Tomato (තක්කාලි) - Target Spot',
    'Tomato (තක්කාලි) - Yellow Leaf Curl Virus',
    'Tomato (තක්කාලි) - Mosaic virus',
    'Tomato (තක්කාලි) - Healthy (නිරෝගී)'
]

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🌱 AI Crop Disease Detection")
st.write("වගාවේ රෝගී වූ පත්‍රයක ඡායාරූපයක් මෙතැනට ඇතුළත් කරන්න.")

uploaded_file = st.file_uploader("ඡායාරූපය තෝරන්න...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='ඔබ ලබා දුන් ඡායාරූපය', use_container_width=True)
    
    st.write("විශ්ලේෂණය කරමින් පවතී...")
    processed_image = preprocess_image(image)
    
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_disease = class_names[predicted_class_index]
    
    if confidence > 75.0:
        st.success(f"**හඳුනාගත් රෝගය:** {predicted_disease}")
        st.info(f"**නිවැරදි වීමේ සම්භාවිතාව:** {confidence:.2f}%")
    else:
        st.warning("⚠️ මට මේ ඡායාරූපය හරියටම හඳුනාගන්න අපහසුයි. කරුණාකර රෝගය සහිත පත්‍රයේ වඩාත් පැහැදිලි ඡායාරූපයක් ඇතුළත් කරන්න.")
