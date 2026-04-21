import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Model එක Load කිරීම
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('crop_disease_model.h5')
    return model

model = load_model()

# 2. ලෙඩ වර්ග වල නම් (PlantVillage එකේ තියෙන නම් ටික මෙතනට දෙන්න ඕනේ)
# උදාහරණයක් විදියට විතරයි මේ නම් ටික දාලා තියෙන්නේ. ඔයාගේ මොඩල් එකට අදාළව මේක වෙනස් කරන්න.
class_names = [
    'Apple - Apple scab', 'Apple - Black rot', 'Apple - Cedar apple rust', 'Apple - healthy',
    'Corn - Cercospora leaf spot', 'Corn - Common rust', 'Corn - healthy',
    'Potato - Early blight', 'Potato - Late blight', 'Potato - healthy',
    'Tomato - Bacterial spot', 'Tomato - Early blight', 'Tomato - healthy'
]

# 3. පින්තූරය AI එකට තේරෙන විදියට හැදීම
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0 # Rescaling (Train කරද්දි කරපු දේම)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 4. Streamlit App එකේ පෙනුම හදමු (UI)
st.title("🌱 AI Crop Disease Detection")
st.write("වගාවේ රෝගී වූ පත්‍රයක ඡායාරූපයක් මෙතැනට ඇතුළත් කරන්න.")

# ෆොටෝ එක අප්ලෝඩ් කරන තැන
uploaded_file = st.file_uploader("ඡායාරූපය තෝරන්න (Choose an image)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # අප්ලෝඩ් කරපු ෆොටෝ එක පෙන්නන්න
    image = Image.open(uploaded_file)
    st.image(image, caption='ඔබ ලබා දුන් ඡායාරූපය', use_container_width=True)
    
    st.write("විශ්ලේෂණය කරමින් පවතී...")
    
    # පින්තූරය හදලා Model එකට දීම
    processed_image = preprocess_image(image)
    
    # ලෙඩේ මොකක්ද කියලා අනුමාන කිරීම (Prediction)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    predicted_disease = class_names[predicted_class_index]
    
    # ප්‍රතිඵලය පෙන්වීම
    st.success(f"**හඳුනාගත් රෝගය:** {predicted_disease}")
    st.info(f"**නිවැරදි වීමේ සම්භාවිතාව (Confidence):** {confidence:.2f}%")
