# 🪙 YOLO Münzzähler

Eine Streamlit-Web-App, die mit einem selbst trainierten **YOLOv8-Modell** Schweizer Münzen auf Fotos erkennt, zählt und den **Gesamtbetrag in Franken** berechnet.

Du kannst ein Bild hochladen oder direkt mit der Kamera ein Foto aufnehmen. Die App zeichnet die erkannten Münzen ein und listet sie nach Münzart mit Anzahl und Teilbetrag auf.

---

## 📑 Inhaltsverzeichnis

- [Funktionen](#-funktionen)
- [Das Modell](#-das-modell)
- [Voraussetzungen](#-voraussetzungen)
- [Projektstruktur](#-projektstruktur)
- [Installation mit uv (empfohlen)](#-installation-mit-uv-empfohlen)
- [Installation mit pip (Alternative)](#-installation-mit-pip-alternative)
- [GPU-Beschleunigung](#-gpu-beschleunigung)
- [App starten](#-app-starten)
- [Bedienung](#-bedienung)
- [Einstellungen im Detail](#-einstellungen-im-detail)
- [Anpassungen](#-anpassungen)
- [Deployment (Streamlit Community Cloud)](#-deployment-streamlit-community-cloud)
- [Fehlerbehebung](#-fehlerbehebung)
- [Tipps für bessere Ergebnisse](#-tipps-für-bessere-ergebnisse)

---

## ✨ Funktionen

- 📁 **Bild hochladen** (`.jpg`, `.jpeg`, `.png`) oder 📷 **Foto mit der Kamera aufnehmen**
- 🔍 **Münzerkennung** mit einem eigenen YOLOv8-Modell (`best.pt`)
- 💰 **Automatische Summenberechnung** für alle erkannten Münzen
- 📊 **Aufschlüsselung** nach Münzart mit Anzahl und Teilbetrag
- 🎚️ **Einstellbarer Konfidenz-Schwellenwert** über die Sidebar
- ⚡ **Automatische Geräteauswahl:** CUDA (NVIDIA) → MPS (Apple Silicon) → CPU

### Unterstützte Münzen

| Klasse | Wert |
|---|---|
| `5 Rappen` | 0.05 Fr. |
| `10 Rappen` | 0.10 Fr. |
| `20 Rappen` | 0.20 Fr. |
| `50 Rappen` | 0.50 Fr. |
| `1 Franken` | 1.00 Fr. |
| `2 Franken` | 2.00 Fr. |
| `5 Franken` | 5.00 Fr. |

> ⚠️ Die Klassennamen des Modells müssen **exakt** den Schlüsseln im `COIN_VALUES`-Dictionary in `app.py` entsprechen. Klassen, die dort nicht vorkommen, werden zwar erkannt, zählen aber mit 0.00 Fr. und werden in der Auflistung entsprechend markiert.

---

## 🧠 Das Modell

Die mitgelieferte Datei `best.pt` (ca. 6 MB) ist ein mit Ultralytics trainiertes Objekterkennungsmodell.

| Eigenschaft | Wert |
|---|---|
| **Aufgabe** | Objekterkennung (`detect`) |
| **Basismodell** | `yolov8n.pt` (YOLOv8 Nano, klein und schnell) |
| **Trainings-Epochen** | 80 |
| **Trainiert mit** | Ultralytics 8.4.155 auf Apple Silicon (`mps`) |
| **Anzahl Klassen** | 7 |

**Klassen im Modell:**

| ID | Klasse |
|---|---|
| 0 | `1 Franken` |
| 1 | `10 Rappen` |
| 2 | `2 Franken` |
| 3 | `20 Rappen` |
| 4 | `5 Franken` |
| 5 | `5 Rappen` |
| 6 | `50 Rappen` |

Diese Namen stimmen mit den Schlüsseln in `COIN_VALUES` überein. Du kannst sie jederzeit selbst prüfen:

```bash
uv run python -c "from ultralytics import YOLO; print(YOLO('best.pt').names)"
```

> ℹ️ Ultralytics-Modelle stehen unter der **AGPL-3.0**-Lizenz. Beachte das, wenn du die App öffentlich anbietest oder weitergibst.

---

## 🧰 Voraussetzungen

| Komponente | Anforderung |
|---|---|
| **Python** | Version **3.12 oder neuer** (laut `pyproject.toml`) |
| **Paketmanager** | [uv](https://docs.astral.sh/uv/) (empfohlen) oder pip |
| **Modell** | `best.pt` im Projektordner |
| **Betriebssystem** | Windows, macOS oder Linux |
| **Hardware** | Läuft auf der CPU, schneller mit NVIDIA-GPU (CUDA) oder Apple Silicon (MPS) |
| **Kamera** | Optional, nur für die Funktion „Foto aufnehmen“ |
| **Linux-Systembibliotheken** | Siehe `packages.txt` (nur nötig auf Linux ohne Desktop, z. B. Server, Docker, Cloud) |

---

## 📂 Projektstruktur

```text
app-coin-counter/
├── app.py              # Die Streamlit-App
├── best.pt             # Trainiertes YOLOv8-Modell
├── pyproject.toml      # Projektdefinition und Abhängigkeiten (uv)
├── uv.lock             # Exakt festgelegte Paketversionen (reproduzierbar)
├── packages.txt        # System-Pakete für Linux / Streamlit Cloud (apt)
└── README.md           # Diese Datei
```

### Die Konfigurationsdateien im Überblick

| Datei | Zweck |
|---|---|
| `pyproject.toml` | Definiert den Projektnamen (`app-coin-counter`), verlangt Python ≥ 3.12 und listet die direkten Abhängigkeiten (`streamlit`, `ultralytics`). |
| `uv.lock` | Hält die exakten Versionen aller Pakete fest, auch der indirekten wie `torch`, `torchvision`, `numpy`, `opencv-python` und `pillow`. So läuft die App bei allen identisch. |
| `packages.txt` | Listet Linux-Systembibliotheken (`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1`), die OpenCV braucht. Streamlit Community Cloud installiert sie automatisch per `apt`. |

> 💡 **Hinweis zu SAHI:** Die App nutzt SAHI nicht mehr. Falls `sahi` in deiner `pyproject.toml` noch unter `dependencies` steht, kannst du es entfernen: `uv remove sahi`. Das spart Installationszeit und hält die Umgebung schlank.

---

## ⚡ Installation mit uv (empfohlen)

[uv](https://docs.astral.sh/uv/) ist ein schneller Python-Paketmanager. Er liest `pyproject.toml` und `uv.lock` und richtet die virtuelle Umgebung samt Python-Version automatisch ein.

### 1. uv installieren

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternativ geht auch `pip install uv`. Prüfen kannst du die Installation mit:

```bash
uv --version
```

### 2. Projektordner vorbereiten

Lege `app.py`, `best.pt`, `pyproject.toml`, `uv.lock` und `packages.txt` in einen gemeinsamen Ordner und wechsle hinein:

```bash
cd app-coin-counter
```

### 3. Abhängigkeiten installieren

```bash
uv sync
```

Dieser Befehl:

- legt automatisch eine virtuelle Umgebung in `.venv/` an,
- lädt bei Bedarf Python 3.12 herunter,
- installiert alle Pakete exakt in den Versionen aus `uv.lock`.

> ⏳ Der erste Durchlauf kann einige Minuten dauern, weil PyTorch ein grosses Paket ist.

### 4. Systembibliotheken installieren (nur Linux)

Auf einem Linux-System ohne grafische Oberfläche (Server, Container, WSL) fehlen oft Bibliotheken, die OpenCV benötigt. Typisches Symptom: `ImportError: libGL.so.1: cannot open shared object file`.

Installiere sie aus `packages.txt`:

```bash
sudo apt-get update
sudo apt-get install -y $(cat packages.txt)
```

Auf **macOS** und **Windows** ist dieser Schritt nicht nötig.

### 5. Installation prüfen

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Die Ausgabe zeigt, welches Gerät die App verwenden kann. Wenn beide Werte `False` sind, läuft die App auf der CPU. Das funktioniert, ist aber langsamer.

---

## 🐍 Installation mit pip (Alternative)

Wenn du uv nicht verwenden möchtest, geht es auch klassisch mit einer virtuellen Umgebung (Python 3.12 oder neuer erforderlich).

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

Dann die Pakete installieren:

```bash
pip install streamlit ultralytics
```

`ultralytics` bringt `torch`, `torchvision`, `numpy`, `opencv-python` und `pillow` automatisch mit. Beachte, dass pip dabei **nicht** die Versionen aus `uv.lock` verwendet, sondern die jeweils aktuellsten. Für exakt reproduzierbare Ergebnisse ist uv besser geeignet.

---

## 🖥️ GPU-Beschleunigung

Die App wählt das Gerät beim Start automatisch:

```python
if torch.cuda.is_available():
    DEVICE = "cuda"      # NVIDIA-GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"       # Apple Silicon
else:
    DEVICE = "cpu"
```

| System | Was du tun musst |
|---|---|
| 🍎 **Apple Silicon (M1 – M5)** | Nichts. MPS wird automatisch erkannt. |
| 🐧 **Linux mit NVIDIA-GPU** | In der Regel nichts. Die Standard-PyTorch-Pakete enthalten CUDA. Ein aktueller NVIDIA-Treiber ist nötig. |
| 🪟 **Windows mit NVIDIA-GPU** | Die Standardinstallation liefert oft nur die CPU-Version von PyTorch. Installiere die CUDA-Version: Den passenden Befehl für dein System erzeugt [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). |
| 💻 **Ohne GPU** | Nichts. Die App läuft auf der CPU. Das Modell ist klein (YOLOv8 Nano), daher ist das meist ausreichend schnell. |

Beispiel für Windows mit uv (die CUDA-Version `cu128` kann bei dir abweichen):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ Ein späteres `uv sync` kann diese Installation wieder durch die Standardversion ersetzen. Starte die App danach mit `uv run --no-sync streamlit run app.py`.

---

## 🚀 App starten

**Mit uv:**

```bash
uv run streamlit run app.py
```

**Mit pip (virtuelle Umgebung aktiv):**

```bash
streamlit run app.py
```

Der Browser öffnet sich automatisch. Falls nicht, rufe diese Adresse auf:

```text
http://localhost:8501
```

Zum Beenden drückst du im Terminal `Ctrl + C`.

### Zugriff vom Smartphone im selben WLAN

Wenn du die Kamera deines Handys nutzen möchtest, kannst du die App im lokalen Netzwerk erreichbar machen:

```bash
uv run streamlit run app.py --server.address 0.0.0.0
```

Rufe dann auf dem Handy `http://<IP-DEINES-COMPUTERS>:8501` auf.

> ℹ️ Browser erlauben den Kamerazugriff meist nur über `localhost` oder über eine verschlüsselte `https`-Verbindung. Falls die Kamera auf dem Handy nicht funktioniert, nutze stattdessen den Bild-Upload. Dort kannst du auf dem Handy ebenfalls ein Foto aufnehmen und hochladen.

---

## 🖱️ Bedienung

1. **Konfidenz einstellen:** Passe in der Sidebar links den Schwellenwert an.
2. **Bildquelle wählen:** Wähle zwischen „📁 Bild hochladen“ und „📷 Foto aufnehmen“.
3. **Bild bereitstellen:** Lade ein Foto hoch oder fotografiere die Münzen.
4. **Ergebnis ansehen:** Links siehst du das Originalbild, rechts das Bild mit den erkannten Münzen.
5. **Auswertung lesen:** Unter den Bildern stehen der Gesamtbetrag und die Aufteilung nach Münzart.

Beispiel für die Ausgabe:

```text
💰 Erkannter Gesamtbetrag: 4.30 Fr.

Aufteilung der erkannten Münzen:
• 3x 1 Franken (Wert: 3.00 Fr.)
• 2x 50 Rappen (Wert: 1.00 Fr.)
• 1x 20 Rappen (Wert: 0.20 Fr.)
• 2x 5 Rappen (Wert: 0.10 Fr.)
```

---

## 🎚️ Einstellungen im Detail

### Konfidenz-Schwellenwert

Legt fest, wie sicher sich das Modell mindestens sein muss, damit eine Erkennung gezählt wird. Der Regler reicht von 0.1 bis 1.0, der Startwert ist 0.25.

| Wert | Wirkung |
|---|---|
| **Niedrig** (z. B. 0.1 – 0.25) | Mehr Erkennungen, aber auch mehr Fehlerkennungen |
| **Mittel** (z. B. 0.4 – 0.6) | Guter Kompromiss für die meisten Fälle |
| **Hoch** (z. B. 0.7 – 1.0) | Nur sehr sichere Erkennungen, dafür können Münzen übersehen werden |

---

## 🔧 Anpassungen

### Eigene Münzklassen oder Währung

Passe das Dictionary `COIN_VALUES` in `app.py` an. Die Schlüssel müssen den Klassennamen in deinem trainierten Modell entsprechen.

```python
COIN_VALUES = {
    "5 Rappen": 0.05,
    "10 Rappen": 0.10,
    # ...
}
```

### Anderes Modell verwenden

Ändere in `app.py` die Variable `MODEL_PATH`:

```python
MODEL_PATH = "pfad/zu/deinem/modell.pt"
```

### Gerät fest vorgeben

Standardmässig wird das beste verfügbare Gerät gewählt. Du kannst es in `app.py` auch fest setzen:

```python
DEVICE = "cpu"   # oder "cuda" bzw. "mps"
```

---

## ☁️ Deployment (Streamlit Community Cloud)

Die App lässt sich auf der [Streamlit Community Cloud](https://streamlit.io/cloud) veröffentlichen:

1. Lade `app.py`, `best.pt`, `pyproject.toml`, `uv.lock`, `packages.txt` und `README.md` in ein GitHub-Repository.
2. Wähle dort **New app**, das Repository und `app.py` als Hauptdatei.
3. Die Plattform installiert die Python-Pakete aus `pyproject.toml` / `uv.lock` und die Systembibliotheken aus `packages.txt` automatisch.

> ⚠️ Auf der Cloud steht in der Regel nur die **CPU** zur Verfügung. Das kleine Nano-Modell läuft dort aber problemlos. Ausserdem gilt für `st.camera_input`: Die Kamera funktioniert im Browser, weil die Cloud-Adresse über `https` läuft.

---

## 🩹 Fehlerbehebung

| Problem | Mögliche Ursache und Lösung |
|---|---|
| `❌ Fehler beim Laden des Modells` | `best.pt` fehlt oder liegt nicht neben `app.py`. Prüfe den Pfad in `MODEL_PATH`. |
| `ImportError: libGL.so.1: cannot open shared object file` | Auf Linux fehlen Systembibliotheken für OpenCV. Installiere sie mit `sudo apt-get install -y $(cat packages.txt)`. |
| `ModuleNotFoundError: No module named ...` | Die Umgebung ist nicht aktiv oder `uv sync` wurde nicht ausgeführt. Starte die App mit `uv run streamlit run app.py`. |
| `requires-python`-Fehler bei der Installation | Deine Python-Version ist älter als 3.12. uv lädt die passende Version automatisch, mit pip musst du Python selbst aktualisieren. |
| `torch.cuda.is_available()` ist `False` trotz NVIDIA-GPU | Es ist die CPU-Version von PyTorch installiert (häufig unter Windows). Installiere die CUDA-Version von [pytorch.org](https://pytorch.org/get-started/locally/) und aktualisiere deinen NVIDIA-Treiber. |
| MPS-Fehler auf dem Mac | Aktualisiere PyTorch und ultralytics (`uv lock --upgrade && uv sync`). Als Notlösung setze `DEVICE = "cpu"`. |
| Keine Münzen erkannt | Senke den Konfidenz-Schwellenwert, verbessere Licht und Bildqualität oder prüfe, ob das Modell auf ähnlichen Bildern trainiert wurde. |
| Falsche Münzen erkannt | Erhöhe den Konfidenz-Schwellenwert. |
| Kamera funktioniert nicht | Erlaube dem Browser den Kamerazugriff. Auf anderen Geräten im Netzwerk geht das oft nur über `https`. Nutze alternativ den Bild-Upload. |
| Klasse „nicht im COIN_VALUES Dictionary definiert“ | Der Klassenname des Modells stimmt nicht mit einem Schlüssel in `COIN_VALUES` überein. Gleiche die Schreibweise an. |
| Port 8501 ist belegt | Starte mit einem anderen Port: `uv run streamlit run app.py --server.port 8502` |

---

## 📸 Tipps für bessere Ergebnisse

- 💡 **Gutes, gleichmässiges Licht** ohne harte Schatten oder Spiegelungen auf den Münzen
- 🎨 **Einfarbiger, kontrastreicher Hintergrund** (z. B. ein dunkles Tuch oder helles Papier)
- 📐 **Von oben fotografieren**, möglichst senkrecht zur Unterlage
- ↔️ **Münzen nicht übereinanderlegen**, damit keine verdeckt wird
- 🔎 **Nah genug herangehen**, damit die Münzen im Bild gross genug sind
- 🧪 **Schwellenwert ausprobieren:** Die Konfidenz lässt sich live in der Sidebar anpassen

---

## 🧱 Verwendete Technologien

| Technologie | Version (laut `uv.lock`) | Zweck |
|---|---|---|
| [Streamlit](https://streamlit.io/) | 1.66.0 | Weboberfläche |
| [Ultralytics YOLOv8](https://docs.ultralytics.com/) | 8.4.182 | Objekterkennung |
| [PyTorch](https://pytorch.org/) | 2.14.1 | Deep-Learning-Backend |
| [OpenCV](https://opencv.org/) | 5.0.0.93 | Bildverarbeitung (von Ultralytics genutzt) |
| [Pillow](https://python-pillow.org/) | 12.3.0 | Bildverarbeitung in der App |
| [NumPy](https://numpy.org/) | 2.5.4 | Numerische Berechnungen |
| [uv](https://docs.astral.sh/uv/) | – | Paket- und Umgebungsverwaltung |