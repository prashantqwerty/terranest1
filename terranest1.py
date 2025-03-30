import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision import models, transforms


def load_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model


def extract_features(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image)
    return features


def suggest_decor(image, text, model):
    features = extract_features(image, model)
    return f"Our AI recommends: {text} with contemporary furniture and smart lighting solutions. (Technical details: Features extracted - {features.shape})"


# Load model
model = load_model()

# Set page config
st.set_page_config(
    page_title="TerraNest | AI Interior Design",
    page_icon="◈",
    layout="centered"
)

# Simplified CSS for readability
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }

    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }

    h1 {
        font-weight: 600;
        margin-bottom: 1rem;
    }

    h2, h3 {
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .stButton>button {
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 500;
    }

    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea,
    .stFileUploader>div>div {
        border: 1px solid #ddd;
        border-radius: 6px;
    }

    hr {
        border: 0.5px solid #eee;
        margin: 1.5rem 0;
    }

    .block-container {
        padding: 2rem 1rem;
    }

    /* List styling */
    ul {
        padding-left: 1.2rem;
    }

    li {
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation
page = st.sidebar.radio("Menu", ["Home", "Services", "Design Studio"])

if page == "Home":
    st.title("TerraNest AI Interior Design")
    st.markdown("""
    Transform your space with our AI-powered design solutions. 
    We combine algorithmic precision with human expertise to create 
    living spaces that reflect modern sensibilities.
    """)

    st.markdown("---")
    st.markdown("### Our Approach")
    st.markdown("""
    - **Data-Driven Design**: Machine learning analyzes thousands of successful interiors
    - **Personalized Solutions**: Tailored to your space dimensions and lifestyle
    - **Sustainable Choices**: Eco-friendly materials prioritized
    - **Real-Time Visualization**: See your design come to life instantly
    """)

elif page == "Services":
    st.title("Our Services")

    st.markdown("""
    ### Complete Space Design
    Full-service interior design from concept to completion.
    Our AI generates multiple layout options which our human designers refine.
    """)

    st.markdown("""
    ### Design Consultation
    60-minute sessions with our design experts to review AI-generated concepts
    and select finishes, furniture, and lighting.
    """)

    st.markdown("""
    ### Virtual Staging
    For real estate professionals: digitally furnish empty spaces to help buyers
    visualize potential.
    """)

elif page == "Design Studio":
    st.title("AI Design Studio")
    st.markdown("""
    Upload a photo of your space and describe your style preferences
    to receive instant design recommendations.
    """)

    uploaded_file = st.file_uploader("Upload room photo (optional)", type=["jpg", "jpeg", "png"])
    user_text = st.text_area("Describe your style (e.g., 'modern minimalist with warm wood tones')")

    if st.button("Generate Design Ideas"):
        if user_text:
            if uploaded_file:
                image = Image.open(uploaded_file)
                suggestion = suggest_decor(image, user_text, model)
            else:
                suggestion = f"Based on your preference for '{user_text}', we recommend clean-lined furniture in neutral tones with strategic accent pieces."

            st.markdown("---")
            st.markdown("### Your Custom Recommendations")
            st.markdown(suggestion)

            st.markdown("""
            #### Suggested Elements:
            - Furniture: Modular sectional in performance fabric
            - Lighting: Smart dimmable fixtures
            - Flooring: Engineered hardwood in warm oak
            - Accents: Textured throw pillows in earthy tones
            """)
        else:
            st.warning("Please describe your style preferences to generate recommendations")