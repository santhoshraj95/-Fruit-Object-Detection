"""
streamlit_retail_app_advanced.py

Advanced Smart Retail Streamlit App (Detection + Billing) with:
- YOLOv8 detection (ultralytics)
- PDF invoice generation (reportlab) with barcode (Code128) + QR
- SQLite database for invoices & inventory
- Inventory dashboard (sales, stock levels)
- Real-time webcam billing (OpenCV)
- Multiple image upload, editable price list, confidence slider

Install dependencies:
pip install streamlit ultralytics opencv-python-headless reportlab pandas openpyxl pillow qrcode

Run:
streamlit run streamlit_retail_app_advanced.py
"""

import io
import os
import uuid
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import qrcode

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.barcode import code128
from reportlab.lib.units import mm

import streamlit as st
import requests
import time
from streamlit_lottie import st_lottie

st.markdown("""
<style>

/* App background */
.stApp{
    background: linear-gradient(120deg,#fff7f0,#ffe6d5);
}

</style>
""", unsafe_allow_html=True)


# -------------------
# Load Lottie Animation
# -------------------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.title("🛒 Smart Retail —  Billing & Inventory")

# Lottie URLs (Fruit Style Animations)
apple_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
banana_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_5ngs2ksb.json")
orange_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_0yfsb3a1.json")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Apple 🍎")
    st_lottie(apple_lottie, height=200)

with col2:
    st.subheader("Banana 🍌")
    st_lottie(banana_lottie, height=200)

with col3:
    st.subheader("Orange 🍊")
    st_lottie(orange_lottie, height=200)




# Try ultralytics (YOLO)
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

# ---------------------------
# Config & Paths
# ---------------------------
MODEL_PATH_DEFAULT = "best.pt"
DB_PATH = "retail.db"
CURRENCY = "₹"
DEFAULT_PRICE = {"apple": 30.0, "banana": 10.0, "orange": 25.0}

# ---------------------------
# SQLite DB Utilities
# ---------------------------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    # invoices table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS invoices (
        id TEXT PRIMARY KEY,
        date TEXT,
        customer TEXT,
        total REAL,
        data BLOB
    )
    """)
    # invoice_items table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS invoice_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_id TEXT,
        item TEXT,
        qty INTEGER,
        unit_price REAL,
        amount REAL,
        FOREIGN KEY(invoice_id) REFERENCES invoices(id)
    )
    """)
    # inventory table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        item TEXT PRIMARY KEY,
        stock INTEGER,
        unit_price REAL
    )
    """)
    conn.commit()
    return conn

db_conn = init_db()
db_cursor = db_conn.cursor()

def seed_inventory(price_map, conn=db_conn):
    # Initialize inventory rows if not present 
    for item, price in price_map.items():
        db_cursor.execute("SELECT item FROM inventory WHERE item=?", (item,))
        if db_cursor.fetchone() is None:
            db_cursor.execute("INSERT INTO inventory(item, stock, unit_price) VALUES (?, ?, ?)",
                              (item, 100, float(price)))
    conn.commit()

def update_inventory_on_sale(cart, conn=db_conn):
    for item, qty in cart.items():
        db_cursor.execute("SELECT stock FROM inventory WHERE item=?", (item,))
        row = db_cursor.fetchone()
        if row:
            new_stock = max(0, row[0] - qty)
            db_cursor.execute("UPDATE inventory SET stock=? WHERE item=?", (new_stock, item))
    conn.commit()

def get_inventory_df(conn=db_conn):
    df = pd.read_sql_query("SELECT item, stock, unit_price FROM inventory", conn)
    return df

def save_invoice_to_db(inv_id, date, customer, total, inv_df, conn=db_conn):
    # store excel bytes as BLOB for reference
    excel_buff = io.BytesIO()
    with pd.ExcelWriter(excel_buff, engine="openpyxl") as writer:
        inv_df.to_excel(writer, index=False, sheet_name="Invoice")
    excel_bytes = excel_buff.getvalue()
    db_cursor.execute("INSERT INTO invoices(id, date, customer, total, data) VALUES (?, ?, ?, ?, ?)",
                      (inv_id, date, customer, float(total), excel_bytes))
    for _, row in inv_df.iterrows():
        db_cursor.execute("INSERT INTO invoice_items(invoice_id, item, qty, unit_price, amount) VALUES (?, ?, ?, ?, ?)",
                          (inv_id, row['item'], int(row['qty']), float(row['unit_price']), float(row['amount'])))
    db_conn.commit()

# ---------------------------
# ReportLab Invoice (with Code128 & QR)
# ---------------------------
def generate_invoice_pdf_bytes(items_rows, total_amount, customer_name="Customer", invoice_id=None):
    if invoice_id is None:
        invoice_id = str(uuid.uuid4())[:8]
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []
    # Title
    story.append(Paragraph("<b>Smart Retail - Fruit Billing</b>", styles['Title']))
    story.append(Spacer(1, 8))
    # Metadata
    meta = f"Customer: {customer_name} &nbsp;&nbsp;&nbsp; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(meta, styles['Normal']))
    story.append(Spacer(1, 8))
    # Barcode (Code128) with invoice id
    barcode = code128.Code128(invoice_id, barHeight=15*mm, barWidth=0.5)
    story.append(barcode)
    story.append(Spacer(1, 8))
    # Table
    table_data = [["Item", "Qty", "Unit Price", "Amount"]]
    for row in items_rows:
        table_data.append([str(row[0]), str(row[1]), f"{row[2]:.2f}", f"{row[3]:.2f}"])
    table_data.append(["", "", "Total", f"{total_amount:.2f} {CURRENCY}"])
    table = Table(table_data, colWidths=[180, 60, 100, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.Color(0.1,0.5,0.8)),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    # QR code (encoded summary)
    qr_text = f"Invoice:{invoice_id}|Total:{total_amount:.2f}{CURRENCY}"
    qr_img = qrcode.make(qr_text)
    qr_buff = io.BytesIO()
    qr_img.save(qr_buff, format="PNG")
    qr_buff.seek(0)
    rl_img = RLImage(qr_buff, width=60, height=60)
    story.append(rl_img)
    story.append(Spacer(1, 12))
    story.append(Paragraph("<i>Thank you for shopping with us!</i>", styles['Normal']))
    doc.build(story)
    buff.seek(0)
    return invoice_id, buff.read()

# ---------------------------
# Detection Utilities
# ---------------------------
@st.cache_resource
def load_model(path):
    if not HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics not installed. pip install ultralytics")
    return YOLO(path)

def pil_to_bgr(img_pil):
    img = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def detect_and_annotate(model, img_bgr, conf=0.4):
    results = model(img_bgr, conf=conf)[0]
    counts = {}
    img = img_bgr.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        confs = float(box.conf[0])
        name = results.names[cls]
        counts[name] = counts.get(name, 0) + 1
        label = f"{name} {confs:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (20,200,20), 2)
        cv2.putText(img, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,200,20), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return counts, img_rgb

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Smart Retail Advanced", layout="wide")


# Sidebar settings
st.sidebar.header("Configuration")
use_model = st.sidebar.checkbox("Use YOLO model for auto-detection", value=True)
model_path = st.sidebar.text_input("Model path", value=MODEL_PATH_DEFAULT)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.95, 0.45, 0.05)

# Price list editor
st.sidebar.markdown("### Price list (editable)")
if "price_map" not in st.session_state:
    st.session_state.price_map = DEFAULT_PRICE.copy()
price_df = pd.DataFrame([{"item": k, "unit_price": v} for k, v in st.session_state.price_map.items()])
edited = st.sidebar.data_editor(price_df, num_rows="dynamic")
# update price_map
new_prices = {}
for _, r in edited.iterrows():
    if r["item"]:
        new_prices[str(r["item"])] = float(r["unit_price"])
st.session_state.price_map = new_prices
# ---------------------------
# Inventory Editor (Sidebar)
# ---------------------------
st.sidebar.markdown("### 📦 Inventory Editor")

inventory_df = get_inventory_df()

edited_inventory = st.sidebar.data_editor(
    inventory_df,
    num_rows="dynamic",
    key="inventory_editor"
)

if st.sidebar.button("Update Inventory"):
    for _, row in edited_inventory.iterrows():
        db_cursor.execute(
            "UPDATE inventory SET stock=?, unit_price=? WHERE item=?",
            (int(row["stock"]),float(row["unit_price"]), row["item"])
        )
    db_conn.commit()
    st.sidebar.success("Inventory updated successfully!")

# Seed inventory table from price_map (if new)
seed_inventory(st.session_state.price_map)

# Load model
model = None
if use_model:
    try:
        model = load_model(model_path)
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        use_model = False
        model = None

# Main layout tabs
tabs = st.tabs(["Retail (Billing)", "Inventory Dashboard", "Invoices", "Sales Analytics"])
with tabs[0]:
    st.header("Retail — Upload or Capture & Bill")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader("Upload images (multiple allowed)", type=["jpg","jpeg","png"], accept_multiple_files=True)
        camera_img = st.camera_input("Or take a photo")

        detections_aggregate = {}
        annotated_list = []

        if uploaded_files:
            for up in uploaded_files:
                pil = Image.open(up)
                bgr = pil_to_bgr(pil)
                if use_model and model:
                    counts, annotated = detect_and_annotate(model, bgr, conf=conf_threshold)
                else:
                    counts = {}
                    annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                for k, v in counts.items():
                    detections_aggregate[k] = detections_aggregate.get(k, 0) + v
                annotated_list.append((up.name, annotated))

        if camera_img:
            pil = Image.open(camera_img)
            bgr = pil_to_bgr(pil)
            if use_model and model:
                counts, annotated = detect_and_annotate(model, bgr, conf=conf_threshold)
            else:
                counts = {}
                annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for k, v in counts.items():
                detections_aggregate[k] = detections_aggregate.get(k, 0) + v
            annotated_list.append(("camera.jpg", annotated))

        # show annotated
        for name, img in annotated_list:
            st.subheader(name)
            st.image(img, use_container_width=True)

        st.markdown("---")
        st.header("Manual Adjustments")
        if "cart" not in st.session_state:
            st.session_state.cart = {}

        st.write("Auto-detected counts (this run):")
        if detections_aggregate:
            for item, qty in detections_aggregate.items():
                st.write(f"{item}: {qty}")
                if st.button(f"Add +1 {item}", key=f"add_{item}"):
                    st.session_state.cart[item] = st.session_state.cart.get(item, 0) + 1
        else:
            st.write("No detections.")

        st.write("Add item manually:")
        col_a, col_b = st.columns(2)
        with col_a:
            add_item = st.text_input("Item name", value="")
            add_qty = st.number_input("Quantity", value=1, min_value=1, step=1)
        with col_b:
            if st.button("Add to cart"):
                name = add_item.strip()
                if name:
                    st.session_state.cart[name] = st.session_state.cart.get(name, 0) + int(add_qty)
                    st.success(f"Added {add_qty} x {name}")

        if detections_aggregate:
            if st.button("Add all detections to cart"):
                for k, v in detections_aggregate.items():
                    st.session_state.cart[k] = st.session_state.cart.get(k, 0) + v
                st.success("Added detected items to cart")

    with col2:
        st.header("Cart & Checkout")
        if "cart" not in st.session_state:
            st.session_state.cart = {}

        # Build cart df
        cart_rows = []
        for k, v in st.session_state.cart.items():
            unit = float(st.session_state.price_map.get(k, DEFAULT_PRICE.get(k, 0.0)))
            cart_rows.append({"item": k, "qty": int(v), "unit_price": unit})
        cart_df = pd.DataFrame(cart_rows)
        if not cart_df.empty:
            cart_df["amount"] = cart_df["qty"] * cart_df["unit_price"]
            edited_cart = st.data_editor(cart_df[["item","qty","unit_price"]], num_rows="dynamic")
            # update session cart & price_map from edited
            new_cart = {}
            for _, r in edited_cart.iterrows():
                if r["item"]:
                    new_cart[str(r["item"])] = int(r["qty"])
                    st.session_state.price_map[str(r["item"])] = float(r["unit_price"])
            st.session_state.cart = new_cart

            # invoice preview
            inv_rows = []
            for it, q in st.session_state.cart.items():
                unit = float(st.session_state.price_map.get(it, DEFAULT_PRICE.get(it, 0.0)))
                amt = unit * q
                inv_rows.append([it, q, unit, amt])
            inv_df = pd.DataFrame(inv_rows, columns=["item","qty","unit_price","amount"])
            total_amount = inv_df["amount"].sum()
            st.write("### Invoice Preview")
            st.dataframe(inv_df.style.format({"unit_price":"{:.2f}","amount":"{:.2f}"}))
            st.markdown(f"**Total: {total_amount:.2f} {CURRENCY}**")
        else:
            inv_df = pd.DataFrame(columns=["item","qty","unit_price","amount"])
            total_amount = 0.0
            st.info("Cart is empty.")

        cust_name = st.text_input("Customer name", value="")
        cust_phone = st.text_input("Customer phone (optional)", value="")

        if st.button("Checkout & Generate Invoice"):
            if inv_df.empty:
                st.error("Cart empty — cannot generate invoice.")
            else:
                inv_id = str(uuid.uuid4())[:8]
                invoice_id, pdf_bytes = generate_invoice_pdf_bytes(inv_rows, float(total_amount), customer_name=cust_name or "Customer", invoice_id=inv_id)
                # save to DB
                save_invoice_to_db(invoice_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cust_name or "Customer", float(total_amount), inv_df)
                # update inventory
                update_inventory_on_sale(st.session_state.cart)
                st.success(f"Invoice {invoice_id} generated and saved.")
                st.download_button("Download Invoice PDF", data=pdf_bytes, file_name=f"invoice_{invoice_id}.pdf", mime="application/pdf")
                # also offer excel download
                excel_buff = io.BytesIO()
                with pd.ExcelWriter(excel_buff, engine="openpyxl") as writer:
                    inv_df.to_excel(writer, index=False, sheet_name="Invoice")
                excel_buff.seek(0)
                st.download_button("Download Invoice Excel", data=excel_buff.read(), file_name=f"invoice_{invoice_id}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                # clear cart if selected
                if st.checkbox("Clear cart after checkout", value=True):
                    st.session_state.cart = {}

with tabs[1]:
    st.header("Inventory Dashboard")
    inv_df = get_inventory_df()
    st.subheader("Current Inventory")
    st.dataframe(inv_df.style.format({"unit_price":"{:.2f}"}))
    st.markdown("---")
    st.subheader("Sales Overview (Last invoices)")
    invoices_df = pd.read_sql_query("SELECT id, date, customer, total FROM invoices ORDER BY date DESC LIMIT 20", db_conn)
    st.dataframe(invoices_df)
    # simple low-stock alert
    low_stock = inv_df[inv_df['stock'] <= 10]
    if not low_stock.empty:
        st.warning("Low stock items detected:")
        st.table(low_stock)

with tabs[2]:
    st.header("Invoices")
    st.subheader("Saved Invoices")
    invs = pd.read_sql_query("SELECT id, date, customer, total FROM invoices ORDER BY date DESC", db_conn)
    st.dataframe(invs)
    sel = st.selectbox("Select invoice to view/download", options=invs['id'].tolist() if not invs.empty else [])
    if sel:
        # fetch invoice blob
        cur = db_conn.cursor()
        cur.execute("SELECT data, date, customer, total FROM invoices WHERE id=?", (sel,))
        row = cur.fetchone()
        if row:
            excel_blob, date_s, cust, total = row
            st.write(f"Invoice ID: {sel}  Date: {date_s}  Customer: {cust}  Total: {total}")
            st.download_button("Download Invoice (Excel)", data=excel_blob, file_name=f"invoice_{sel}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------
# Sales Analytics
# ---------------------------
with tabs[3]:
    st.header("📊 Sales Analytics")

    sales_df = pd.read_sql_query(
        "SELECT date, total FROM invoices",
        db_conn
    )

    if not sales_df.empty:

        sales_df["date"] = pd.to_datetime(sales_df["date"])
        sales_df["day"] = sales_df["date"].dt.date

        daily_sales = sales_df.groupby("day")["total"].sum().reset_index()

        st.subheader("Daily Sales Trend")
        st.line_chart(daily_sales.set_index("day"))

        st.subheader("Sales Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Revenue",
                f"₹ {sales_df['total'].sum():.2f}"
            )

        with col2:
            st.metric(
                "Average Invoice",
                f"₹ {sales_df['total'].mean():.2f}"
            )

    else:
        st.info("No sales data available yet.")
# -------------------
# Load Lottie
# -------------------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

loader_lottie = load_lottie_url(
    "https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json"
)
# -------------------
# Loader Screen
# -------------------
with st.spinner("Loading Smart Fruit Store... 🍎"):
    st_lottie(loader_lottie, height=100)
    time.sleep(3)

st.success("Welcome to Smart Fruit Store 🍊")
st.caption("Advanced Smart Retail — includes barcode in invoices, persistent DB, inventory dashboard. For production, run on a reliable server & hook scales/POS APIs.")


import streamlit as st
import requests
from streamlit_lottie import st_lottie

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

delivery_lottie = load_lottie_url(
    "https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json"
)

# -------------------
# Hero Section
# -------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("""
        <h1 style='color:#ff4d4d; font-size:48px;'>
        Fresh Fruits Delivered to Your Door 🍎
        </h1>
        <p style='font-size:20px;'>
        Fast • Fresh • Affordable
        </p>
    """, unsafe_allow_html=True)

with col2:
    st_lottie(delivery_lottie, height=300)











