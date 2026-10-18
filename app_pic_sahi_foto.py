import os
import streamlit as st
from PIL import Image
import numpy as np

# --- SAHI IMPORTS ---
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# ============================================================
# 1. STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="YOLOv8 + SAHI Münzzähler",
    layout="centered"
)

st.title("🪙 YOLOv8 + SAHI Münzzähler")
st.write(
    "Lade ein Bild hoch oder fotografiere Münzen direkt mit der Kamera."
)


# ============================================================
# 2. MODELL
# ============================================================

MODEL_PATH = "best.pt"



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

use_sahi = st.sidebar.checkbox(
    "SAHI aktivieren (Slicing)",
    value=True
)

slice_size = st.sidebar.slider(
    "Kachelgröße (Slice Size)",
    min_value=128,
    max_value=640,
    value=256,
    step=32
)

overlap_ratio = st.sidebar.slider(
    "Überlappung (Overlap Ratio)",
    min_value=0.0,
    max_value=0.5,
    value=0.2,
    step=0.05
)

conf_threshold = st.sidebar.slider(
    "Konfidenz-Schwellenwert",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)


# ============================================================
# 5. MODELL LADEN
# ============================================================

@st.cache_resource
def load_sahi_model(model_path, conf_thresh):

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_thresh,
        device="mps"
    )


try:
    sahi_model = load_sahi_model(
        MODEL_PATH,
        conf_threshold
    )

    st.success("✅ Modell erfolgreich geladen!")

except Exception as e:

    st.error(
        f"❌ Fehler beim Laden des Modells: {e}"
    )

    st.stop()


# ============================================================
# 6. BILDQUELLE AUSWÄHLEN
# ============================================================

st.subheader("📷 Bild auswählen")

input_method = st.radio(
    "Wie möchtest du das Bild bereitstellen?",
    [
        "📁 Bild hochladen",
        "📷 Foto aufnehmen"
    ],
    horizontal=True
)


uploaded_file = None


# ------------------------------------------------------------
# OPTION 1: BILD HOCHLADEN
# ------------------------------------------------------------

if input_method == "📁 Bild hochladen":

    uploaded_file = st.file_uploader(
        "Wähle ein Bild aus...",
        type=["jpg", "jpeg", "png"]
    )


# ------------------------------------------------------------
# OPTION 2: FOTO AUFNEHMEN
# ------------------------------------------------------------

else:

    uploaded_file = st.camera_input(
        "📷 Münzen fotografieren"
    )


# ============================================================
# 7. BILD VERARBEITEN
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.divider()

    col1, col2 = st.columns(2)


    # ========================================================
    # ORIGINALBILD
    # ========================================================

    with col1:

        st.subheader("Originalbild")

        st.image(
            image,
            use_container_width=True
        )


    # ========================================================
    # OBJEKTERKENNUNG
    # ========================================================

    with col2:

        st.subheader("Objektkennung")

        with st.spinner(
            "🔍 Modell analysiert das Bild..."
        ):

            img_np = np.array(image)


            # =================================================
            # SAHI
            # =================================================

            if use_sahi:

                sahi_result = get_sliced_prediction(
                    image=img_np,
                    detection_model=sahi_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio
                )


                # Temporärer Ausgabeordner

                export_dir = "./tmp_sahi"

                os.makedirs(
                    export_dir,
                    exist_ok=True
                )


                # Visualisierung speichern

                sahi_result.export_visuals(
                    export_dir=export_dir,
                    file_name="result",
                    text_size=1,
                    rect_th=2
                )


                result_path = (
                    f"{export_dir}/result.png"
                )


                res_image = Image.open(
                    result_path
                )


                st.image(
                    res_image,
                    use_container_width=True
                )


                detections = (
                    sahi_result.object_prediction_list
                )


            # =================================================
            # NORMALES YOLO
            # =================================================

            else:

                pure_yolo = sahi_model.model

                results = pure_yolo.predict(
                    source=image,
                    device="mps",
                    conf=conf_threshold
                )

                res = results[0]

                res_plotted = res.plot()

                res_image = Image.fromarray(
                    res_plotted[..., ::-1]
                )

                st.image(
                    res_image,
                    use_container_width=True
                )

                detections = res.boxes


    # ========================================================
    # 8. MÜNZEN AUSWERTEN
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Auswertung & Gesamtsumme"
    )


    total_amount = 0.0

    detected_coins_summary = {}


    # ========================================================
    # KEINE MÜNZEN
    # ========================================================

    if len(detections) == 0:

        st.info(
            "Keine Münzen erkannt."
        )


    # ========================================================
    # MÜNZEN GEFUNDEN
    # ========================================================

    else:

        # ----------------------------------------------------
        # SAHI
        # ----------------------------------------------------

        if use_sahi:

            for det in detections:

                class_name = det.category.name

                detected_coins_summary[class_name] = (
                    detected_coins_summary.get(
                        class_name,
                        0
                    ) + 1
                )

                total_amount += COIN_VALUES.get(
                    class_name,
                    0.0
                )


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        else:

            for box in detections:

                class_id = int(
                    box.cls[0]
                )

                class_name = (
                    sahi_model.model.names[
                        class_id
                    ]
                )

                detected_coins_summary[class_name] = (
                    detected_coins_summary.get(
                        class_name,
                        0
                    ) + 1
                )

                total_amount += COIN_VALUES.get(
                    class_name,
                    0.0
                )


        # ====================================================
        # GESAMTBETRAG
        # ====================================================

        st.metric(
            label="💰 Erkannter Gesamtbetrag",
            value=f"{total_amount:.2f} Fr."
        )


        # ====================================================
        # DETAILAUFLISTUNG
        # ====================================================

        st.write(
            "**Aufteilung der erkannten Münzen:**"
        )


        for coin, count in detected_coins_summary.items():

            single_value = COIN_VALUES.get(
                coin,
                0.0
            )

            subtotal = (
                single_value * count
            )


            if single_value > 0:

                st.write(
                    f"• **{count}x** {coin} "
                    f"(Wert: {subtotal:.2f} Fr.)"
                )

            else:

                st.write(
                    f"• **{count}x** {coin} "
                    f"*(Klasse nicht im "
                    f"COIN_VALUES Dictionary definiert!)*"
                )
