import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="YOLO Münzzähler",
    layout="centered"
)

st.title("🪙 YOLO Münzzähler")
st.write(
    "Lade ein Bild hoch oder fotografiere Münzen direkt mit der Kamera."
)


# ============================================================
# 2. MODELL & GERÄT
# ============================================================

MODEL_PATH = "best.pt"

# CUDA wenn verfügbar, sonst MPS (Apple Silicon) wenn verfügbar, sonst CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ============================================================
# 3. MÜNZWERTE
# ============================================================

COIN_VALUES = {
    "5 Rappen": 0.05,
    "10 Rappen": 0.10,
    "20 Rappen": 0.20,
    "50 Rappen": 0.50,
    "1 Franken": 1.00,
    "2 Franken": 2.00,
    "5 Franken": 5.00,
}


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.header("🔧 Einstellungen")

conf_threshold = st.sidebar.slider(
    "Konfidenz-Schwellenwert",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "Überlappungs-Schwellenwert (IoU)",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Ab dieser Überlappung wird nur die Box mit der höheren Konfidenz behalten. "
         "Kleiner = aggressiveres Entfernen."
)


# ============================================================
# 5. MODELL LADEN
# ============================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


try:
    model = load_model(MODEL_PATH)
    st.success("✅ Modell erfolgreich geladen!")
except Exception as e:
    st.error(f"❌ Fehler beim Laden des Modells: {e}")
    st.stop()


# ============================================================
# 6. BILDQUELLE AUSWÄHLEN
# ============================================================

st.subheader("📷 Bild auswählen")

input_method = st.radio(
    "Wie möchtest du das Bild bereitstellen?",
    ["📁 Bild hochladen", "📷 Foto aufnehmen"],
    horizontal=True
)

uploaded_file = None

if input_method == "📁 Bild hochladen":
    uploaded_file = st.file_uploader(
        "Wähle ein Bild aus...",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input("📷 Münzen fotografieren")


# ============================================================
# 7. BILD VERARBEITEN
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Originalbild")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Objekterkennung")

        with st.spinner("🔍 Modell analysiert das Bild..."):
            results = model.predict(
                source=image,
                device=DEVICE,
                conf=conf_threshold,
                iou=iou_threshold,
                agnostic_nms=True,
                verbose=False
            )

            res = results[0]

            # plot() liefert ein BGR-Array -> in RGB umkehren
            res_image = Image.fromarray(res.plot()[..., ::-1])

            st.image(res_image, use_container_width=True)

            detections = res.boxes


    # ========================================================
    # 8. MÜNZEN AUSWERTEN
    # ========================================================

    st.divider()
    st.subheader("📊 Auswertung & Gesamtsumme")

    total_amount = 0.0
    detected_coins_summary = {}

    if len(detections) == 0:
        st.info("Keine Münzen erkannt.")

    else:
        detected_coins_summary = {}
        confidences = {}  # Klasse -> Liste der Konfidenzen
        for box in detections:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])

            detected_coins_summary[class_name] = (
                detected_coins_summary.get(class_name, 0) + 1
            )
            confidences.setdefault(class_name, []).append(conf)

            total_amount += COIN_VALUES.get(class_name, 0.0)

        st.metric(
            label="💰 Erkannter Gesamtbetrag",
            value=f"{total_amount:.2f} Fr."
        )

        st.write("**Aufteilung der erkannten Münzen:**")

        for coin, count in detected_coins_summary.items():
            single_value = COIN_VALUES.get(coin, 0.0)
            subtotal = single_value * count

            coin_confs = confidences[coin]
            avg_conf = sum(coin_confs) / len(coin_confs)
            conf_list = ", ".join(f"{c:.0%}" for c in coin_confs)

            if single_value > 0:
                st.write(
                    f"• **{count}x** {coin} (Wert: {subtotal:.2f} Fr.) "
                    f"– Ø Konfidenz: **{avg_conf:.0%}** "
                    f"*(einzeln: {conf_list})*"
                )
            else:
                st.write(
                    f"• **{count}x** {coin} "
                    f"*(Klasse nicht im COIN_VALUES Dictionary definiert!)*"
                )