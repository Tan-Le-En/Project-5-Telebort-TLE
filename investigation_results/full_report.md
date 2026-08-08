# Data Investigation -- Full Report

*Generated 2026-08-08 11:04*

Detection Settings: `scaleFactor=1.1, minNeighbors=5`

## 1. Summary of Tested Conditions

| # | Photo File | Condition / Label | Boxes Found | True Positives | False Positives | Missed | Detection Rate | Notes |
|---|------------|-------------------|-------------|----------------|-----------------|--------|----------------|-------|
| 1 | `outing.jpg` | **outing** | 18 | 18 | 0 | 0 | 100.0% | Face detected cleanly |
| 2 | `01_normal_sunny.jpg` | **normal_sunny** | 1 | 1 | 0 | 0 | 100.0% | Bright lighting condition |
| 3 | `02_darkness.jpg` | **darkness** | 0 | 0 | 0 | 1 | 0.0% | Low contrast lighting affected detection |
| 4 | `03_light_pointed_at_face.jpg` | **light_pointed_at_face** | 1 | 1 | 0 | 0 | 100.0% | Face detected cleanly |
| 5 | `04_very_sunny.jpg` | **very_sunny** | 1 | 1 | 0 | 0 | 100.0% | Bright lighting condition |
| 6 | `05_suny_rays.jpg` | **suny_rays** | 0 | 0 | 0 | 1 | 0.0% | No face detected |
| 7 | `capture_01_condition_01.jpg` | **capture_01_condition_01** | 2 | 2 | 0 | 0 | 100.0% | Face detected cleanly |
| 8 | `capture_02_face_down.jpg` | **capture_02_face_down** | 2 | 2 | 0 | 0 | 100.0% | Face detected cleanly |
| 9 | `capture_03_face_up.jpg` | **capture_03_face_up** | 0 | 0 | 0 | 1 | 0.0% | No face detected |
| 10 | `capture_04_bright_sun.jpg` | **capture_04_bright_sun** | 1 | 1 | 0 | 0 | 100.0% | Bright lighting condition |
| 11 | `capture_05_darkness.jpg` | **capture_05_darkness** | 2 | 2 | 0 | 0 | 100.0% | Detected despite low light |

**Best performing condition:** `outing.jpg` (outing) with 18 detected face(s)
**Worst performing condition:** `02_darkness.jpg` (darkness) with 0 detected face(s)

## 2. Key Findings & Analysis

- **Frontal Geometry**: Haar Cascades rely on pre-trained rigid facial features (eyes, nose, mouth alignment). Direct frontal faces yield optimal detection accuracy.
- **Lighting & Contrast**: Evenly lit environments yield crisp feature boundaries. Low-contrast or severe shadows can cause missed detections.
- **Hyperparameter Balance**: `scaleFactor=1.1` and `minNeighbors=5` balance detection sensitivity and false positive suppression.

## 3. Annotated Image Artifacts

- **outing**: Saved to `investigation_results/01_outing_detected.jpg`
- **normal_sunny**: Saved to `investigation_results/02_01_normal_sunny_detected.jpg`
- **darkness**: Saved to `investigation_results/03_02_darkness_detected.jpg`
- **light_pointed_at_face**: Saved to `investigation_results/04_03_light_pointed_at_face_detected.jpg`
- **very_sunny**: Saved to `investigation_results/05_04_very_sunny_detected.jpg`
- **suny_rays**: Saved to `investigation_results/06_05_suny_rays_detected.jpg`
- **capture_01_condition_01**: Saved to `investigation_results/07_capture_01_condition_01_detected.jpg`
- **capture_02_face_down**: Saved to `investigation_results/08_capture_02_face_down_detected.jpg`
- **capture_03_face_up**: Saved to `investigation_results/09_capture_03_face_up_detected.jpg`
- **capture_04_bright_sun**: Saved to `investigation_results/10_capture_04_bright_sun_detected.jpg`
- **capture_05_darkness**: Saved to `investigation_results/11_capture_05_darkness_detected.jpg`
