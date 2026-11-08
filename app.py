import streamlit as st
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from ultralytics import YOLO

# HEIC-Support für iPhone-Fotos aktivieren
register_heif_opener()

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
# 2. MODELL, GERÄT & GRENZWERTE
# ============================================================

MODEL_PATH = "best.pt"

# Muss identisch zum Training sein (IMG_SIZE im Trainingsskript)
TRAIN_IMGSZ = 960
PREDICT_IMGSZ = TRAIN_IMGSZ

# Münzgrösse in Pixeln, NACHDEM das Bild auf PREDICT_IMGSZ skaliert wurde.
# Werte aus der Messung der Trainingsdaten bei imgsz=960:
# 10%-Perzentil = 60 px, Median = 109 px, 90%-Perzentil = 186 px
MIN_COIN_PX = 100
MAX_COIN_PX = 250

# Unter dieser Konfidenz wird eine Erkennung als unsicher markiert
LOW_CONF = 0.60

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

agnostic = st.sidebar.checkbox(
    "Klassen-unabhängige NMS (agnostic)",
    value=True,
    help="An: pro Position bleibt nur eine Klasse. "
         "Aus: überlappende Boxen mit verschiedenen Klassen bleiben erhalten (gut zum Debuggen)."
)

st.sidebar.caption(
    f"Modell-Auflösung: {PREDICT_IMGSZ} px\n\n"
    f"Min. Münzgrösse: {MIN_COIN_PX} px\n\n"
    f"Unsicher unter: {LOW_CONF:.0%}"
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
except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
    st.error(f"❌ Fehler beim Laden des Modells: {e}")
    st.stop()


# ============================================================
# 6. BILDQUELLE AUSWÄHLEN
# ============================================================

st.subheader("📷 Bild auswählen")

st.info(
    "💡 Tipp für beste Ergebnisse: Münzen auf einen hellen, einfarbigen "
    "Untergrund legen, von oben fotografieren (Rückkamera) und genug nah "
    "herangehen, sodass die Münzen gut sichtbar sind."
)

input_method = st.radio(
    "Wie möchtest du das Bild bereitstellen?",
    ["📁 Bild hochladen", "📷 Foto aufnehmen"],
    horizontal=True
)

uploaded_file = None

if input_method == "📁 Bild hochladen":
    uploaded_file = st.file_uploader(
        "Wähle ein Bild aus...",
        type=["jpg", "jpeg", "png", "heic", "heif"]
    )
else:
    uploaded_file = st.camera_input("📷 Münzen fotografieren")


# ============================================================
# 7. BILD VERARBEITEN
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    image = image.convert("RGB")

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
                imgsz=PREDICT_IMGSZ,
                device=DEVICE,
                conf=conf_threshold,
                iou=iou_threshold,
                agnostic_nms=agnostic,
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
        confidences = {}   # Klasse -> Liste der Konfidenzen
        small_coins = []   # (Nr., Klasse, Pixelgrösse)
        large_coins = []   # (Nr., Klasse, Pixelgrösse)
        unsure_coins = []  # (Nr., Klasse, Konfidenz)

        # So stark wurde das Bild für das Modell verkleinert/vergrössert
        img_w, img_h = image.size
        scale = PREDICT_IMGSZ / max(img_w, img_h)

        for i, box in enumerate(detections, start=1):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])

            # Boxgrösse in Pixeln, so wie das Modell sie sieht
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            coin_px = max(x2 - x1, y2 - y1) * scale

            if coin_px < MIN_COIN_PX:
                small_coins.append((i, class_name, coin_px))
            elif coin_px > MAX_COIN_PX:
                large_coins.append((i, class_name, coin_px))

            if conf < LOW_CONF:
                unsure_coins.append((i, class_name, conf))

            detected_coins_summary[class_name] = (
                detected_coins_summary.get(class_name, 0) + 1
            )
            confidences.setdefault(class_name, []).append(conf)

            total_amount += COIN_VALUES.get(class_name, 0.0)

        # ---- Warnungen (vor dem Gesamtbetrag anzeigen) ----
        if small_coins:
            details = ", ".join(
                f"#{i} ({name}, ca. {px:.0f} px)" for i, name, px in small_coins
            )
            st.warning(
                f"⚠️ {len(small_coins)} Münze(n) sind zu klein für eine "
                f"zuverlässige Erkennung (Minimum: {MIN_COIN_PX} px): {details}. "
                f"Bitte näher an die Münzen herangehen oder das Bild zuschneiden."
            )

        if large_coins:
            details = ", ".join(
                f"#{i} ({name}, ca. {px:.0f} px)" for i, name, px in large_coins
            )
            st.warning(
                f"⚠️ {len(large_coins)} Münze(n) sind ungewöhnlich gross "
                f"(über {MAX_COIN_PX} px): {details}. "
                f"Das Modell kennt solche Grössen kaum, bitte etwas weiter weg fotografieren."
            )

        if unsure_coins:
            details = ", ".join(
                f"#{i} ({name}, {c:.0%})" for i, name, c in unsure_coins
            )
            st.warning(
                f"⚠️ {len(unsure_coins)} Erkennung(en) sind unsicher "
                f"(Konfidenz unter {LOW_CONF:.0%}): {details}. "
                f"Bitte das Ergebnis kontrollieren."
            )

        # ---- Ergebnis ----
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