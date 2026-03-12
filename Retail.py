import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# Try YOLO
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except:
    HAS_ULTRALYTICS = False

st.set_page_config(page_title="Smart Fruit Billing", layout="wide")

st.title("🛒 Smart Fruit Store — Billing")

# ---------------------------
# Default Prices
# ---------------------------
DEFAULT_PRICE = {
    "apple": 30,
    "banana": 10,
    "orange": 25
}

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Price List")

if "price_map" not in st.session_state:
    st.session_state.price_map = DEFAULT_PRICE.copy()

price_df = pd.DataFrame(
    [{"item":k,"price":v} for k,v in st.session_state.price_map.items()]
)

edited = st.sidebar.data_editor(price_df)

new_price = {}
for _,r in edited.iterrows():
    new_price[r["item"]] = r["price"]

st.session_state.price_map = new_price

# ---------------------------
# Load YOLO model
# ---------------------------
model = None

if HAS_ULTRALYTICS:
    try:
        model = YOLO("best.pt")
    except:
        st.warning("Model not found")

# ---------------------------
# Image Upload
# ---------------------------
uploaded = st.file_uploader(
    "Upload fruit image",
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

camera = st.camera_input("Take Photo")

detections = {}

def detect(img):
    counts={}
    if model:
        results = model(img)[0]

        for box in results.boxes:
            cls=int(box.cls[0])
            name = results.names[cls]

            counts[name] = counts.get(name,0)+1
    return counts

# ---------------------------
# Run Detection
# ---------------------------
if uploaded:

    for file in uploaded:

        img = Image.open(file)
        img_np = np.array(img)

        st.image(img)

        counts = detect(img_np)

        for k,v in counts.items():
            detections[k] = detections.get(k,0)+v

if camera:

    img = Image.open(camera)
    img_np = np.array(img)

    st.image(img)

    counts = detect(img_np)

    for k,v in counts.items():
        detections[k] = detections.get(k,0)+v

# ---------------------------
# Cart
# ---------------------------
if "cart" not in st.session_state:
    st.session_state.cart = {}

st.subheader("Detected Fruits")

for item,qty in detections.items():

    st.write(f"{item} : {qty}")

    if st.button(f"Add {item}"):

        st.session_state.cart[item] = \
            st.session_state.cart.get(item,0)+qty

# ---------------------------
# Sidebar Checkout
# ---------------------------
st.sidebar.header("Checkout")

rows=[]

for item,qty in st.session_state.cart.items():

    price = st.session_state.price_map.get(item,0)

    rows.append({
        "item":item,
        "qty":qty,
        "price":price,
        "amount":qty*price
    })

cart_df = pd.DataFrame(rows)

if not cart_df.empty:

    total = cart_df["amount"].sum()

    st.sidebar.dataframe(cart_df)

    st.sidebar.markdown(f"### Total : ₹ {total}")

    if st.sidebar.button("Clear Cart"):

        st.session_state.cart={}

else:

    st.sidebar.write("Cart empty")
