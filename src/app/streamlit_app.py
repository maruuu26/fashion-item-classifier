import io
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st

from src.models.baseline import build_resnet18
from src.data.fashion_mnist import get_transforms

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat"
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@st.cache_resource
def load_model(ckpt_path="models/checkpoints/best.pt", device="cpu"):
    model = build_resnet18(num_classes=10, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def preprocess(img: Image.Image, img_size=224):
    _, eval_tfms = get_transforms(img_size=img_size)
    return eval_tfms(img).unsqueeze(0)

def main():
    st.title("Fashion Classifier")
    st.write("Upload a clothing image and get the top-3 predictions.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(device=device)

    uploaded = st.file_uploader("Choose an image", type=["png", 'jpg', "jpeg"])
    if uploaded:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(img, caption="Your image", width=256)

        x = preprocess(img).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        topk = probs.argsort()[-3:][::-1]
        st.subheader("Top-3 Predictions")
        for idx in topk:
            st.write(f"{CLASS_NAMES[idx]} - {probs[idx]:.2%}")

if __name__ == "__main__":
    main()