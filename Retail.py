"""
retail_billing_app.py

Minimized Smart Retail Billing App:
- Upload images OR use webcam
- YOLOv8 detection (optional)
- Sidebar: editable price list + checkout
- No invoice generation, no DB, no extra tabs

Install:
    pip install streamlit ultralytics opencv-python-headless pillow

Run:
    streamlit run retail_billing_app.py
"""

import io
import numpy as np
import streamlit as st
from PIL import Image
import cv2

# ── Optional YOLO ──────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH_DEFAULT = "best.pt"
CURRENCY = "₹"
DEFAULT_PRICES = {"apple": 30.0, "banana": 10.0, "orange": 25.0}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Retail – Billing", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(120deg, #fff7f0, #ffe6d5); }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Smart Retail — Billing")

# ── Session state ──────────────────────────────────────────────────────────────
if "price_map" not in st.session_state:
    st.session_state.price_map = DEFAULT_PRICES.copy()
if "cart" not in st.session_state:
    st.session_state.cart = {}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

use_model = st.sidebar.checkbox("Use YOLO auto-detection", value=True)
model_path = st.sidebar.text_input("Model path", value=MODEL_PATH_DEFAULT)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Price List")

import pandas as pd

price_df = pd.DataFrame([
    {"item": k, "unit_price (₹)": v}
    for k, v in st.session_state.price_map.items()
])
edited_prices = st.sidebar.data_editor(price_df, num_rows="dynamic", key="price_editor")

# Sync edited prices back
new_prices = {}
for _, r in edited_prices.iterrows():
    if r["item"]:
        new_prices[str(r["item"])] = float(r["unit_price (₹)"])
st.session_state.price_map = new_prices

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛍️ Checkout")

# Build cart summary in sidebar
cart_rows = []
for item, qty in st.session_state.cart.items():
    unit = float(st.session_state.price_map.get(item, DEFAULT_PRICES.get(item, 0.0)))
    cart_rows.append({"Item": item, "Qty": qty, "Unit ₹": unit, "Amount ₹": round(unit * qty, 2)})

if cart_rows:
    cart_df = pd.DataFrame(cart_rows)
    st.sidebar.dataframe(cart_df, use_container_width=True, hide_index=True)
    total = cart_df["Amount ₹"].sum()
    st.sidebar.markdown(f"**Total: ₹ {total:.2f}**")

    if st.sidebar.button("✅ Confirm & Clear Cart", type="primary"):
        st.sidebar.success(f"Order confirmed! Total: ₹ {total:.2f}")
        st.session_state.cart = {}
else:
    st.sidebar.info("Cart is empty.")

if st.sidebar.button("🗑️ Clear Cart"):
    st.session_state.cart = {}
    st.sidebar.success("Cart cleared.")

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if not HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
    return YOLO(path)

model = None
if use_model:
    try:
        model = load_model(model_path)
    except Exception as e:
        st.sidebar.warning(f"Model not loaded: {e}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def pil_to_bgr(img_pil):
    img = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def detect_and_annotate(mdl, img_bgr, conf=0.4):
    results = mdl(img_bgr, conf=conf)[0]
    counts = {}
    img = img_bgr.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        name = results.names[cls]
        counts[name] = counts.get(name, 0) + 1
        label = f"{name} {confidence:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (20, 200, 20), 2)
        cv2.putText(img, label, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 200, 20), 2)
    return counts, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ── Main area ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📷 Upload or Capture")
    uploaded_files = st.file_uploader(
        "Upload images (multiple allowed)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    camera_img = st.camera_input("Or take a photo")

    detections_aggregate = {}
    annotated_list = []

    sources = [(up.name, Image.open(up)) for up in (uploaded_files or [])]
    if camera_img:
        sources.append(("camera.jpg", Image.open(camera_img)))

    for name, pil in sources:
        bgr = pil_to_bgr(pil)
        if use_model and model:
            counts, annotated = detect_and_annotate(model, bgr, conf=conf_threshold)
        else:
            counts = {}
            annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for k, v in counts.items():
            detections_aggregate[k] = detections_aggregate.get(k, 0) + v
        annotated_list.append((name, annotated))

    for name, img in annotated_list:
        st.caption(name)
        st.image(img, use_container_width=True)

with col_right:
    st.subheader("🔍 Detections")

    if detections_aggregate:
        for item, qty in detections_aggregate.items():
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"**{item}** × {qty}")
            with c2:
                if st.button("Add", key=f"add_{item}"):
                    st.session_state.cart[item] = st.session_state.cart.get(item, 0) + qty
                    st.success(f"+{qty} {item}")

        if st.button("➕ Add ALL to cart", use_container_width=True):
            for k, v in detections_aggregate.items():
                st.session_state.cart[k] = st.session_state.cart.get(k, 0) + v
            st.success("All detections added to cart!")
    else:
        st.info("Upload an image to detect items.")

    st.markdown("---")
    st.subheader("✏️ Add Manually")
    add_item = st.text_input("Item name")
    add_qty = st.number_input("Quantity", min_value=1, value=1, step=1)
    if st.button("Add to cart", use_container_width=True):
        name = add_item.strip()
        if name:
            st.session_state.cart[name] = st.session_state.cart.get(name, 0) + int(add_qty)
            st.success(f"Added {add_qty} × {name}")
        else:
            st.warning("Enter an item name.")
