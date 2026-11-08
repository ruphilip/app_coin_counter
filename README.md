# 🪙 YOLO Coin Counter

A Streamlit web app that uses a custom-trained **YOLOv8 model** to detect and count Swiss coins in photos and calculate the **total amount in Swiss francs**.

You can upload an image or take a photo directly with your camera. The app draws the detected coins onto the image and lists them by coin type with count and subtotal.

---

## 📑 Table of Contents

- [Features](#-features)
- [The Model](#-the-model)
- [Requirements](#-requirements)
- [Project Structure](#-project-structure)
- [Installation with uv (recommended)](#-installation-with-uv-recommended)
- [Installation with pip (alternative)](#-installation-with-pip-alternative)
- [GPU Acceleration](#-gpu-acceleration)
- [Starting the App](#-starting-the-app)
- [Usage](#-usage)
- [Settings in Detail](#-settings-in-detail)
- [Customization](#-customization)
- [Deployment (Streamlit Community Cloud)](#-deployment-streamlit-community-cloud)
- [Troubleshooting](#-troubleshooting)
- [Tips for Better Results](#-tips-for-better-results)

---

## ✨ Features

- 📁 **Upload an image** (`.jpg`, `.jpeg`, `.png`) or 📷 **take a photo with the camera**
- 🔍 **Coin detection** with a custom YOLOv8 model (`best.pt`)
- 💰 **Automatic total calculation** for all detected coins
- 📊 **Breakdown** by coin type with count and subtotal
- 🎚️ **Adjustable confidence threshold** via the sidebar
- ⚡ **Automatic device selection:** CUDA (NVIDIA) → MPS (Apple Silicon) → CPU

### Supported Coins

| Class | Value |
|---|---|
| `5 Rappen` | 0.05 CHF |
| `10 Rappen` | 0.10 CHF |
| `20 Rappen` | 0.20 CHF |
| `50 Rappen` | 0.50 CHF |
| `1 Franken` | 1.00 CHF |
| `2 Franken` | 2.00 CHF |
| `5 Franken` | 5.00 CHF |

> ⚠️ The model's class names must **exactly** match the keys in the `COIN_VALUES` dictionary in `app.py`. Classes that do not appear there are still detected, but count as 0.00 CHF and are marked accordingly in the list.

---

## 🧠 The Model

The included file `best.pt` (about 6 MB) is an object detection model trained with Ultralytics.

| Property | Value |
|---|---|
| **Task** | Object detection (`detect`) |
| **Base model** | `yolov8n.pt` (YOLOv8 Nano, small and fast) |
| **Training epochs** | 80 |
| **Trained with** | Ultralytics 8.4.155 on Apple Silicon (`mps`) |
| **Number of classes** | 7 |

**Classes in the model:**

| ID | Class |
|---|---|
| 0 | `1 Franken` |
| 1 | `10 Rappen` |
| 2 | `2 Franken` |
| 3 | `20 Rappen` |
| 4 | `5 Franken` |
| 5 | `5 Rappen` |
| 6 | `50 Rappen` |

These names match the keys in `COIN_VALUES`. You can verify them yourself at any time:

```bash
uv run python -c "from ultralytics import YOLO; print(YOLO('best.pt').names)"
```

> ℹ️ Ultralytics models are licensed under the **AGPL-3.0**. Keep this in mind if you offer the app publicly or distribute it.

---

## 🧰 Requirements

| Component | Requirement |
|---|---|
| **Python** | Version **3.12 or newer** (per `pyproject.toml`) |
| **Package manager** | [uv](https://docs.astral.sh/uv/) (recommended) or pip |
| **Model** | `best.pt` in the project folder |
| **Operating system** | Windows, macOS or Linux |
| **Hardware** | Runs on the CPU, faster with an NVIDIA GPU (CUDA) or Apple Silicon (MPS) |
| **Camera** | Optional, only needed for the "Take a photo" feature |
| **Linux system libraries** | See `packages.txt` (only needed on Linux without a desktop, e.g. servers, Docker, cloud) |

---

## 📂 Project Structure

```text
app-coin-counter/
├── app.py              # The Streamlit app
├── best.pt             # Trained YOLOv8 model
├── pyproject.toml      # Project definition and dependencies (uv)
├── uv.lock             # Pinned package versions (reproducible)
├── packages.txt        # System packages for Linux / Streamlit Cloud (apt)
└── README.md           # This file
```

### The Configuration Files at a Glance

| File | Purpose |
|---|---|
| `pyproject.toml` | Defines the project name (`app-coin-counter`), requires Python ≥ 3.12 and lists the direct dependencies (`streamlit`, `ultralytics`). |
| `uv.lock` | Records the exact versions of all packages, including indirect ones such as `torch`, `torchvision`, `numpy`, `opencv-python` and `pillow`. This way the app behaves identically for everyone. |
| `packages.txt` | Lists Linux system libraries (`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`) that OpenCV needs. Streamlit Community Cloud installs them automatically via `apt`. |

> 💡 **Note on SAHI:** The app no longer uses SAHI. If `sahi` is still listed under `dependencies` in your `pyproject.toml`, you can remove it: `uv remove sahi`. This saves installation time and keeps the environment lean.

---

## ⚡ Installation with uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. It reads `pyproject.toml` and `uv.lock` and automatically sets up the virtual environment including the Python version.

### 1. Install uv

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively, `pip install uv` also works. You can check the installation with:

```bash
uv --version
```

### 2. Prepare the project folder

Put `app.py`, `best.pt`, `pyproject.toml`, `uv.lock` and `packages.txt` into a common folder and change into it:

```bash
cd app-coin-counter
```

### 3. Install dependencies

```bash
uv sync
```

This command:

- automatically creates a virtual environment in `.venv/`,
- downloads Python 3.12 if needed,
- installs all packages in the exact versions from `uv.lock`.

> ⏳ The first run can take a few minutes because PyTorch is a large package.

### 4. Install system libraries (Linux only)

On a Linux system without a graphical interface (server, container, WSL), libraries needed by OpenCV are often missing. Typical symptom: `ImportError: libGL.so.1: cannot open shared object file`.

Install them from `packages.txt`:

```bash
sudo apt-get update
sudo apt-get install -y $(cat packages.txt)
```

This step is not needed on **macOS** and **Windows**.

### 5. Verify the installation

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

The output shows which device the app can use. If both values are `False`, the app runs on the CPU. That works, but is slower.

---

## 🐍 Installation with pip (alternative)

If you prefer not to use uv, you can also proceed the classic way with a virtual environment (Python 3.12 or newer required).

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Then install the packages:

```bash
pip install streamlit ultralytics
```

`ultralytics` automatically brings in `torch`, `torchvision`, `numpy`, `opencv-python` and `pillow`. Note that pip does **not** use the versions from `uv.lock` here, but the latest ones available. For exactly reproducible results, uv is the better choice.

---

## 🖥️ GPU Acceleration

The app selects the device automatically at startup:

```python
if torch.cuda.is_available():
    DEVICE = "cuda"      # NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"       # Apple Silicon
else:
    DEVICE = "cpu"
```

| System | What you need to do |
|---|---|
| 🍎 **Apple Silicon (M1 – M5)** | Nothing. MPS is detected automatically. |
| 🐧 **Linux with NVIDIA GPU** | Usually nothing. The standard PyTorch packages include CUDA. A current NVIDIA driver is required. |
| 🪟 **Windows with NVIDIA GPU** | The default installation often provides only the CPU version of PyTorch. Install the CUDA version: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) generates the right command for your system. |
| 💻 **No GPU** | Nothing. The app runs on the CPU. The model is small (YOLOv8 Nano), so this is usually fast enough. |

Example for Windows with uv (the CUDA version `cu128` may differ on your machine):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ A later `uv sync` may replace this installation with the default version again. Afterwards, start the app with `uv run --no-sync streamlit run app.py`.

---

## 🚀 Starting the App

**With uv:**

```bash
uv run streamlit run app.py
```

**With pip (virtual environment active):**

```bash
streamlit run app.py
```

The browser opens automatically. If it doesn't, open this address:

```text
http://localhost:8501
```

To stop the app, press `Ctrl + C` in the terminal.

### Access from a smartphone on the same Wi-Fi

If you want to use your phone's camera, you can make the app reachable on the local network:

```bash
uv run streamlit run app.py --server.address 0.0.0.0
```

Then open `http://<YOUR-COMPUTER-IP>:8501` on your phone.

> ℹ️ Browsers usually only allow camera access via `localhost` or over an encrypted `https` connection. If the camera doesn't work on your phone, use the image upload instead. There you can also take a photo on your phone and upload it.

---

## 🖱️ Usage

1. **Set the confidence:** Adjust the threshold in the sidebar on the left.
2. **Choose the image source:** Pick "📁 Bild hochladen" (upload an image) or "📷 Foto aufnehmen" (take a photo).
3. **Provide an image:** Upload a photo or photograph the coins.
4. **View the result:** On the left you see the original image, on the right the image with the detected coins.
5. **Read the evaluation:** Below the images you find the total amount and the breakdown by coin type.

Example output (the app's interface is in German):

```text
💰 Erkannter Gesamtbetrag: 4.30 Fr.

Aufteilung der erkannten Münzen:
• 3x 1 Franken (Wert: 3.00 Fr.)
• 2x 50 Rappen (Wert: 1.00 Fr.)
• 1x 20 Rappen (Wert: 0.20 Fr.)
• 2x 5 Rappen (Wert: 0.10 Fr.)
```

In English: "Detected total amount: 4.30 CHF" followed by the breakdown of detected coins with count and value.

---

## 🎚️ Settings in Detail

### Confidence threshold

Determines how certain the model must be at minimum for a detection to be counted. The slider ranges from 0.1 to 1.0, the default is 0.25.

| Value | Effect |
|---|---|
| **Low** (e.g. 0.1 – 0.25) | More detections, but also more false detections |
| **Medium** (e.g. 0.4 – 0.6) | Good compromise for most cases |
| **High** (e.g. 0.7 – 1.0) | Only very confident detections, but coins may be missed |

---

## 🔧 Customization

### Custom coin classes or currency

Adjust the `COIN_VALUES` dictionary in `app.py`. The keys must match the class names in your trained model.

```python
COIN_VALUES = {
    "5 Rappen": 0.05,
    "10 Rappen": 0.10,
    # ...
}
```

### Using a different model

Change the `MODEL_PATH` variable in `app.py`:

```python
MODEL_PATH = "path/to/your/model.pt"
```

### Setting the device manually

By default, the best available device is chosen. You can also set it explicitly in `app.py`:

```python
DEVICE = "cpu"   # or "cuda" / "mps"
```

---

## ☁️ Deployment (Streamlit Community Cloud)

The app can be published on the [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Upload `app.py`, `best.pt`, `pyproject.toml`, `uv.lock`, `packages.txt` and `README.md` to a GitHub repository.
2. There, choose **New app**, select the repository and `app.py` as the main file.
3. The platform automatically installs the Python packages from `pyproject.toml` / `uv.lock` and the system libraries from `packages.txt`.

> ⚠️ In the cloud, usually only the **CPU** is available. The small Nano model runs there without problems, though. Also note for `st.camera_input`: the camera works in the browser because the cloud address uses `https`.

---

## 🩹 Troubleshooting

| Problem | Possible cause and solution |
|---|---|
| `❌ Fehler beim Laden des Modells` (error loading the model) | `best.pt` is missing or not located next to `app.py`. Check the path in `MODEL_PATH`. |
| `ImportError: libGL.so.1: cannot open shared object file` | System libraries for OpenCV are missing on Linux. Install them with `sudo apt-get install -y $(cat packages.txt)`. |
| `ModuleNotFoundError: No module named ...` | The environment is not active or `uv sync` was not run. Start the app with `uv run streamlit run app.py`. |
| `requires-python` error during installation | Your Python version is older than 3.12. uv downloads the right version automatically; with pip you have to update Python yourself. |
| `torch.cuda.is_available()` is `False` despite an NVIDIA GPU | The CPU version of PyTorch is installed (common on Windows). Install the CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) and update your NVIDIA driver. |
| MPS error on Mac | Update PyTorch and ultralytics (`uv lock --upgrade && uv sync`). As a workaround, set `DEVICE = "cpu"`. |
| No coins detected | Lower the confidence threshold, improve lighting and image quality, or check whether the model was trained on similar images. |
| Wrong coins detected | Raise the confidence threshold. |
| Camera doesn't work | Allow the browser to access the camera. On other devices in the network this often only works via `https`. Alternatively, use the image upload. |
| Class "not defined in the COIN_VALUES dictionary" | The model's class name doesn't match any key in `COIN_VALUES`. Align the spelling. |
| Port 8501 is in use | Start with a different port: `uv run streamlit run app.py --server.port 8502` |

---

## 📸 Tips for Better Results

- 💡 **Good, even lighting** without harsh shadows or reflections on the coins
- 🎨 **Plain, high-contrast background** (e.g. a dark cloth or light paper)
- 📐 **Shoot from above**, as perpendicular to the surface as possible
- ↔️ **Don't stack coins on top of each other**, so none is hidden
- 🔎 **Get close enough** so the coins appear large enough in the image
- 🧪 **Experiment with the threshold:** the confidence can be adjusted live in the sidebar

---

## 🧱 Technologies Used

| Technology | Version (per `uv.lock`) | Purpose |
|---|---|---|
| [Streamlit](https://streamlit.io/) | 1.66.0 | Web interface |
| [Ultralytics YOLOv8](https://docs.ultralytics.com/) | 8.4.182 | Object detection |
| [PyTorch](https://pytorch.org/) | 2.14.1 | Deep learning backend |
| [OpenCV](https://opencv.org/) | 5.0.0.93 | Image processing (used by Ultralytics) |
| [Pillow](https://python-pillow.org/) | 12.3.0 | Image processing in the app |
| [NumPy](https://numpy.org/) | 2.5.4 | Numerical computing |
| [uv](https://docs.astral.sh/uv/) | – | Package and environment management |