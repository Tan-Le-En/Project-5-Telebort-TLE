"""
Camera Diagnostic Tool for OpenCV on Windows / Multi-platform.
Tests available camera indices and backends, checks frame quality (non-black, valid brightness/variance),
saves sample test images, and reports the best working camera setup.
"""

import cv2 as cv
import numpy as np
import os
import sys

def test_cameras():
    print("=" * 60)
    print("      OPENCV CAMERA DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"OpenCV Version: {cv.__version__}")
    print(f"Operating System: {sys.platform}")
    print("-" * 60)

    # List of backends to test
    backends = []
    if hasattr(cv, 'CAP_DSHOW'):
        backends.append(("CAP_DSHOW", cv.CAP_DSHOW))
    if hasattr(cv, 'CAP_MSMF'):
        backends.append(("CAP_MSMF", cv.CAP_MSMF))
    backends.append(("CAP_ANY", cv.CAP_ANY))

    camera_indices = [0, 1, 2, 3]
    results = []

    for idx in camera_indices:
        for backend_name, backend_code in backends:
            print(f"\nTesting Camera Index {idx} with backend {backend_name}...")
            try:
                cap = cv.VideoCapture(idx, backend_code)
            except Exception as e:
                print(f"  --> Exception while opening VideoCapture({idx}, {backend_name}): {e}")
                continue

            if not cap.isOpened():
                print(f"  --> Could not open VideoCapture({idx}, {backend_name})")
                cap.release()
                continue

            # Read a few warmup frames
            success = False
            best_frame = None
            mean_val = 0.0
            std_val = 0.0

            for frame_num in range(10):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    current_mean = float(np.mean(frame))
                    current_std = float(np.std(frame))
                    # Keep frame with best variance / brightness
                    if current_std > std_val or best_frame is None:
                        best_frame = frame
                        mean_val = current_mean
                        std_val = current_std
                        success = True

            cap.release()

            if not success or best_frame is None:
                print(f"  --> Connected, but failed to read valid frames.")
                continue

            h, w, c = best_frame.shape
            is_usable = (mean_val > 5.0) and (std_val > 2.0)
            status_str = "WORKING (Real Image)" if is_usable else "BLANK/DARK"

            print(f"  --> Status: {status_str}")
            print(f"      Resolution: {w}x{h}, Mean Brightness: {mean_val:.2f}, StdDev/Variance: {std_val:.2f}")

            # Save diagnostic image
            filename = f"cam_test_idx{idx}_{backend_name}.jpg"
            cv.imwrite(filename, best_frame)
            print(f"      Saved diagnostic image to: {filename}")

            results.append({
                "index": idx,
                "backend_name": backend_name,
                "backend_code": backend_code,
                "width": w,
                "height": h,
                "mean_brightness": mean_val,
                "std_dev": std_val,
                "usable": is_usable,
                "filename": filename
            })

    print("\n" + "=" * 60)
    print("      DIAGNOSTIC SUMMARY REPORT")
    print("=" * 60)

    working_cameras = [r for r in results if r["usable"]]

    if not results:
        print("No camera devices could be opened.")
    else:
        for r in results:
            status = "SUCCESS" if r["usable"] else "FAILED (Black/Dark Feed)"
            print(f"Index {r['index']} | Backend {r['backend_name']:10s} | {r['width']}x{r['height']} | Mean: {r['mean_brightness']:6.2f} | Std: {r['std_dev']:6.2f} | {status}")

    print("-" * 60)
    best_cam = None
    if working_cameras:
        # Sort working cameras by std_dev (texture/detail) and brightness
        working_cameras.sort(key=lambda x: (x["std_dev"], x["mean_brightness"]), reverse=True)
        best_cam = working_cameras[0]
        print(f"\nRECOMMENDED CAMERA:")
        print(f"  Index: {best_cam['index']}")
        print(f"  Backend: {best_cam['backend_name']} ({best_cam['backend_code']})")
        print(f"  Resolution: {best_cam['width']}x{best_cam['height']}")
        print(f"  Sample Image: {best_cam['filename']}")
    else:
        print("\nNo working camera feed found.")
        print("Troubleshooting steps:")
        print(" 1. Check Windows Privacy & Security -> Camera -> allow app access.")
        print(" 2. Ensure camera shutter/cover is open.")
        print(" 3. Close other apps using the webcam (Zoom, Teams, Chrome, etc.).")

    return best_cam

if __name__ == "__main__":
    test_cameras()
