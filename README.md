# Project 05 — Face Detector

> **🛠️ Stack for this lesson** — Python / OpenCV (Haar Cascades) · runs locally (webcam) and on any image file.
> 📥 Template: [/learn/ai2/template/project-05-face-detector](/learn/ai2/template/project-05-face-detector)

A classical computer vision project that uses pre-trained Haar Cascade XML to find faces — the same algorithm behind every camera auto-focus from 2010 onwards. You'll build it twice: once for a still image, once for a live webcam stream.

**Time:** ~3 hours (image 2h, video 1h) · **Concept:** Concepts 13–17

---

## What You'll Build

Two small Python scripts that load a pre-trained classifier and draw boxes around every face they find. 6 outcome-driven TODOs across two parts:

| Part | Focus | Outcome |
|------|-------|---------|
| 1. Image | Static detection | Faces in `sample.jpg` boxed in green; count printed |
| 2. Video | Live capture | Webcam frames boxed in real time; `q` exits cleanly |

Read each file, infer what the TODO needs, and write it. There are no commented-out answers.

## Run It

```bash
# Install OpenCV
pip install opencv-python

# Download the cascade XML (one-off)
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Image detection
python image.py

# Webcam detection (local only)
python video.py
```

## 🔬 Data Investigation

> **📓 Where this lives now:** open your lesson note for this activity.
> Click **Take Notes** at the top of the lesson page if you don't already have a note —
> it creates one titled `<Lesson title> — <COURSE>` with a scaffold pre-filled from
> the lesson headings. Add an H2 section titled **`Investigation`** to that note and put
> your work there. The note persists across devices and is queryable by the AI Tutor.
>
> 💡 **Tip:** open the AI Tutor while viewing your note and ask
> *"help me draft my Investigation findings for [topic]"*. The Tutor will append a draft to your note;
> you can drag it into the **Investigation** section if it lands elsewhere.

**Prompts to answer in your note:**

Haar Cascades are *fast* but they fail in predictable ways: profile faces, faces with sunglasses, faces under strong shadow, faces tilted past 30°, faces that aren't human. AI assistants will report a count happily — your job is to find the failure modes.

1. **What you checked.** Test with at least 4 images: one well-lit front-facing portrait, one profile shot, one with sunglasses, one with multiple faces at different scales. Vary `scaleFactor` and `minNeighbors` parameters.
2. **What you found.** Where did the cascade miss? Where did it false-positive (e.g. boxes on shadows)?
3. **What you fixed.** Did tuning `scaleFactor=1.1` vs `1.3` change recall? Did `minNeighbors` cut false positives?
4. **What this would mean in production.** If your app auto-tagged photos, what's the cost of missing a face vs. boxing a non-face?

"It detected 3 faces" earns 0 — verify visually.

## Verify

Your project is done when:

- [ ] Haar Cascade XML loads without throwing.
- [ ] `image.py` displays the resized sample with green boxes around each face and prints `Found N faces`.
- [ ] `video.py` opens the local webcam and tracks faces in real time.
- [ ] Pressing `q` cleanly releases the camera and closes windows.
- [ ] You can explain in one sentence why grayscale conversion happens *before* detection.
- [ ] Investigation section in your lesson note covers at least 4 test images.

## Stretch

Pick one and note what you tried in your reflection:
- Load `haarcascade_eye.xml` and detect eyes only inside detected face rectangles.
- Overlay a face count on every webcam frame using `cv.putText`.
- Replace Haar Cascades with the more modern `cv.dnn.readNetFromCaffe` + a SSD face detector and compare false-positive rates.

## 🪞 Reflect on Your Work

Answer in 2-3 sentences each, in this README under your TODO commits. Your tutor reads these as part of grading.

1. **What did you learn that you didn't know before?** Name the most surprising thing — a bug you hit, a syntax quirk, a way the classifier failed on a face you thought was easy.
2. **How did you collaborate with AI?** If you used Claude / ChatGPT / Cursor / Copilot, what part of the work did *you* contribute — the prompt, the verification, the design decision, the bug-fix? If you didn't use AI, what was the hardest thing to figure out alone?
3. **How do you know your code works?** Describe one specific thing you did to confirm — a screenshot of the working detection, a count you double-checked, a parameter you swept.

> AI is a great collaborator. Owning your thinking, verifying the output, and explaining your design choices is what *learning* looks like in this course.

## Submit

When the Verify checklist is green, head to **[/learn/ai2/certification](/learn/ai2/certification)** and submit your scripts plus 2–3 detection screenshots.

<!-- claude-template-fix: readme-v3 -->

<!-- claude-template-fix: notes-migration-v1 -->
