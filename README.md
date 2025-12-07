# fashion-item-classifier
Fashion Item Classifier with Outfit Recommendations


A full end-to-end machine learning web application that classifies clothing items from images and suggests similar items to shop from real retailers.

Built with PyTorch and Streamlit

---

# Features

- Upload any clothing image (PNG, JPG, JPEG)
- Classifies the item into one of 10 clothing categories using a CNN (ResNet18)
- Displays the Top-3 predictions with confidence
- Automatically generates shopping links to:
  - H&M  
  - Zara  
  - Uniqlo  
  - ASOS  
  - Amazon  
- Uses a color-aware search query (e.g., “gray Pullover”)

---

# Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pillow
- Fashion-MNIST Dataset
- Apple Silicon (MPS GPU support)

---

# Project Structure
fashion-classifier/
|
|--- src/
||--- app/streamlit_app.py
||--- data/fashion_mnist.py
||--- models/baseline.py
||___ train/train_classifier.py
├── models/checkpoints/best.pt 
├── runs/ 
├── requirements.txt
└── README.md

# How to run the app locally
1. Clone the repo

2. Create/Activate the environment:
    conda create -n fashionml python=3.10
    conda activate fashionml
    pip install -r requirements.txt

3. Run the app
    streamlit run src/app/streamlit_app.py

