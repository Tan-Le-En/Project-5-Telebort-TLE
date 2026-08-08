# Project 05 — Face Detector: Full Setup Guide
### Image Detection + Video Stretch (Eyes, Smile, Face Counter, FPS)

This is a complete, start-to-finish guide to get `image.py` and `video_stretch.py` running on your machine, even if you've never used Python or OpenCV before.

---

## 1. Check Python is installed

Open a terminal (Command Prompt on Windows, Terminal on Mac) and run:

```bash
python --version
```

If that fails, try:

```bash
python3 --version
```

You need Python 3.8 or newer. If neither command works, install Python from [python.org/downloads](https://www.python.org/downloads/) first, then come back here.

> Windows users: during install, make sure you tick **"Add Python to PATH"** — if you skip this, `python` won't be recognized in the terminal.

---

## 2. Create a project folder

Put everything in one folder so the scripts can find the image and XML files sitting next to them.

```bash
mkdir face-detector
cd face-detector
```

Copy `image.py` and `video_stretch.py` into this folder now.

---

## 3. Install OpenCV

Still inside `face-detector`, run:

```bash
pip install opencv-python
```

If `pip` isn't recognized, try `pip3 install opencv-python` or `python -m pip install opencv-python`.

**Verify it installed correctly:**

```bash
python -c "import cv2; print(cv2.__version__)"
```

You should see a version number (e.g. `4.10.0`) with no errors.

---

## 4. Download the Haar Cascade files

You need **three** XML files in the same folder as your scripts:

| File | Used by |
|------|---------|
| `haarcascade_frontalface_default.xml` | both `image.py` and `video_stretch.py` |
| `haarcascade_eye.xml` | `video_stretch.py` only |
| `haarcascade_smile.xml` | `video_stretch.py` only |

Run all three downloads from inside your `face-detector` folder:

```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml
```

**No `wget` on your system (common on Windows)?** Just open each link in your browser, then use **File → Save As**, and save it into the `face-detector` folder with the exact filename from the URL (don't let the browser rename it to `.txt` or add `(1)`).

**Confirm all three landed in the right place:**

```bash
# Mac/Linux
ls *.xml

# Windows
dir *.xml
```

You should see all three filenames listed.

---

## 5. Add a test photo (for `image.py` only)

`image.py` looks for a file called **`outing.jpg`** in the same folder. Drop any photo with one or more faces into `face-detector` and rename it to `outing.jpg`.

> If your version of the script uses a different filename (some earlier drafts used `sample.jpg`), open `image.py`, find the `cv.imread(...)` line near the top, and match the filename there to whatever you name your photo.
