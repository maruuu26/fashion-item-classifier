import io

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st

from src.models.baseline import build_resnet18
from src.data.fashion_mnist import get_transforms


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def guess_color_name(img: Image.Image) -> str | None:
    """
    Return a rough color name like 'black', 'white', 'gray', 'brown', 'red', 'blue', etc.
    Brown is handled before gray so warm low-sat colors aren't mislabeled.
    """
    # downsample for speed
    small = img.resize((64, 64))
    arr = np.asarray(small).reshape(-1, 3).astype(np.float32)

    # compute mean color in RGB
    r_mean = arr[:, 0].mean()
    g_mean = arr[:, 1].mean()
    b_mean = arr[:, 2].mean()

    # convert to HSV-like values
    r, g, b = r_mean / 255.0, g_mean / 255.0, b_mean / 255.0
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min
    v = c_max                      # value / brightness
    s = 0.0 if c_max == 0 else delta / c_max

    if delta == 0:
        h = 0.0
    else:
        if c_max == r:
            h = 60 * (((g - b) / delta) % 6)
        elif c_max == g:
            h = 60 * (((b - r) / delta) + 2)
        else:
            h = 60 * (((r - g) / delta) + 4)

    # now h ∈ [0,360), s,v ∈ [0,1]

    # 1) very dark / very light
    if v < 0.18:
        return "black"
    if v > 0.92 and s < 0.25:
        return "white"

    # 2) detect "brown" as warm hue, medium brightness, medium-ish saturation
    #    (check this BEFORE gray so warm low-sat colors become brown, not gray)
    if 15 <= h <= 60 and 0.08 <= s <= 0.8 and 0.20 <= v <= 0.8:
        return "brown"

    # 3) gray: low saturation, mid brightness
    if s < 0.18 and 0.2 <= v <= 0.9:
        return "gray"

    # 4) hue-based basic colors
    if (h >= 330) or (h < 20):
        return "red"
    if 20 <= h < 50:
        return "orange"
    if 50 <= h < 70:
        return "yellow"
    if 70 <= h < 160:
        return "green"
    if 160 <= h < 210:
        return "cyan"
    if 210 <= h < 260:
        return "blue"
    if 260 <= h < 300:
        return "purple"
    if 300 <= h < 330:
        return "magenta"

    return None


def retailer_search_urls(query: str) -> dict[str, str]:
    """Build simple search URLs for several retailers."""
    q = query.replace(" ", "%20")
    return {
        "H&M":   f"https://www2.hm.com/en_us/search-results.html?q={q}",
        "Zara":  f"https://www.zara.com/us/en/search?searchTerm={q}",
        "Uniqlo": f"https://www.uniqlo.com/us/en/search?q={q}",
        "ASOS":  f"https://www.asos.com/search/?q={q}",
        "Amazon": f"https://www.amazon.com/s?k={q}",
    }


@st.cache_resource
def load_model(ckpt_path: str = "models/checkpoints/best.pt", device: str = "cpu"):
    model = build_resnet18(num_classes=10, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(img: Image.Image, img_size: int = 224) -> torch.Tensor:
    _, eval_tfms = get_transforms(img_size=img_size)
    return eval_tfms(img).unsqueeze(0)  # [1, C, H, W]


def main():
    st.title("Fashion Classifier")
    st.write("Upload a clothing image and get the top-3 predictions plus shopping links.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(device=device)

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(img, caption="Your image", width=256)

        x = preprocess(img).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # shape (10,)

        # top-3 predicted classes
        topk = probs.argsort()[-3:][::-1]
        st.subheader("Top-3 Predictions")
        for idx in topk:
            st.write(f"{CLASS_NAMES[idx]} — {probs[idx]:.2%}")

        # -----------------------------
        # Shop similar items section
        # -----------------------------
        st.markdown("---")
        st.subheader("Shop similar items")

        # best prediction = first element of topk
        best_idx = int(topk[0])
        best_class = CLASS_NAMES[best_idx]

        # rough color guess from uploaded image
        color = guess_color_name(img)
        if color:
            search_query = f"{color} {best_class}"
        else:
            search_query = best_class

        st.caption(f"Searching for: **{search_query}**")

        urls = retailer_search_urls(search_query)
        for name, url in urls.items():
            st.markdown(f"- [{name}]({url})")


if __name__ == "__main__":
    main()
