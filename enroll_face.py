#!/usr/bin/env python3
"""Enroll a face identity by capturing frames from streamerpi.

Usage:
    python enroll_face.py Dan          # Enroll "Dan" with 15 captures over 10 seconds
    python enroll_face.py Dan --count 20 --interval 0.5  # Custom settings
    python enroll_face.py --list       # List enrolled identities
    python enroll_face.py --delete Dan # Remove an enrollment

Stand in front of the camera and slowly turn your head left/right and
tilt up/down during capture for diverse angle coverage.
"""

import argparse
import json
import os
import sys
import time

import cv2
import httpx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision.face_id import FaceID, DEFAULT_DB_PATH


STREAMERPI_FACE_DB_URL = "https://192.168.1.110:8080/face_db/update"


def push_to_pi():
    """Push the face DB to streamerpi for local real-time face ID."""
    from vision.face_id import DEFAULT_DB_PATH
    if not os.path.exists(DEFAULT_DB_PATH):
        print("No face DB to push.")
        return False

    with open(DEFAULT_DB_PATH, "r") as f:
        data = json.load(f)

    try:
        with httpx.Client(timeout=5.0, verify=False) as client:
            r = client.post(STREAMERPI_FACE_DB_URL, json=data)
            r.raise_for_status()
            result = r.json()
            names = result.get("names", [])
            print(f"Pushed to Pi: {len(names)} identities ({', '.join(names)})")
            return True
    except Exception as e:
        print(f"Failed to push to Pi: {e}")
        print("(You can push manually later with: python enroll_face.py --push)")
        return False

STREAMERPI_CAPTURE_URL = "https://192.168.1.110:8080/capture"


def fetch_frame() -> np.ndarray | None:
    """Fetch a single frame from streamerpi."""
    try:
        with httpx.Client(timeout=5.0, verify=False) as client:
            r = client.get(STREAMERPI_CAPTURE_URL)
            r.raise_for_status()
            arr = np.frombuffer(r.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
    except Exception as e:
        print(f"  Failed to fetch frame: {e}")
        return None


def enroll(name: str, count: int = 15, interval: float = 0.7):
    """Capture frames and enroll a face."""
    print(f"\n=== Enrolling '{name}' ===")
    print(f"Will capture {count} frames at {interval}s intervals ({count * interval:.0f}s total)")
    print("Stand in front of the camera and slowly turn your head...\n")

    fid = FaceID()
    if not fid.init_models():
        print("ERROR: Could not load models")
        return False

    fid.load_db()

    embeddings = []
    skipped = 0

    for i in range(count):
        frame = fetch_frame()
        if frame is None:
            skipped += 1
            continue

        faces = fid.detect_faces(frame)
        if faces is None or len(faces) == 0:
            print(f"  [{i+1}/{count}] No face detected (skipped)")
            skipped += 1
            time.sleep(interval)
            continue

        if len(faces) > 1:
            print(f"  [{i+1}/{count}] Multiple faces ({len(faces)}), using largest")

        # Use the largest face
        largest = max(range(len(faces)), key=lambda j: faces[j][2] * faces[j][3])
        face = faces[largest]
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

        embedding = fid.extract_embedding(frame, face)
        if embedding is None:
            print(f"  [{i+1}/{count}] Embedding extraction failed (skipped)")
            skipped += 1
            time.sleep(interval)
            continue

        embeddings.append(embedding)
        print(f"  [{i+1}/{count}] Face at ({x},{y},{w}x{h}) - embedding captured")

        time.sleep(interval)

    print(f"\nCaptured {len(embeddings)} embeddings ({skipped} skipped)")

    if len(embeddings) < 3:
        print("ERROR: Need at least 3 good embeddings. Try again with better lighting/positioning.")
        return False

    # Check consistency: compute pairwise distances
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = 1.0 - np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            )
            distances.append(d)

    avg_dist = np.mean(distances)
    max_dist = np.max(distances)
    print(f"Embedding consistency: avg distance={avg_dist:.3f}, max={max_dist:.3f}")

    if avg_dist > 0.5:
        print("WARNING: High variance in embeddings. Multiple people may have been captured.")
        resp = input("Continue anyway? [y/N] ")
        if resp.lower() != "y":
            return False

    fid.enroll(name, embeddings)
    print(f"\n'{name}' enrolled successfully with {len(embeddings)} samples")
    print(f"Database: {fid._db_path}")

    # Push to Pi for real-time local face ID
    print("\nPushing embeddings to streamerpi...")
    push_to_pi()

    return True


def list_enrolled():
    """List all enrolled identities."""
    if not os.path.exists(DEFAULT_DB_PATH):
        print("No face database found.")
        return

    with open(DEFAULT_DB_PATH, "r") as f:
        data = json.load(f)

    enrolled = data.get("enrolled", {})
    if not enrolled:
        print("No identities enrolled.")
        return

    print(f"\nEnrolled identities ({len(enrolled)}):")
    for name, embedding in enrolled.items():
        dim = len(embedding)
        print(f"  - {name} ({dim}-dim embedding)")


def delete_enrolled(name: str):
    """Delete an enrolled identity."""
    if not os.path.exists(DEFAULT_DB_PATH):
        print("No face database found.")
        return

    with open(DEFAULT_DB_PATH, "r") as f:
        data = json.load(f)

    enrolled = data.get("enrolled", {})
    if name not in enrolled:
        print(f"'{name}' not found in database.")
        return

    del enrolled[name]
    data["enrolled"] = enrolled

    with open(DEFAULT_DB_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Deleted '{name}' from face database.")


def test_identify():
    """Quick test: capture one frame and try to identify faces."""
    fid = FaceID()
    if not fid.init_models():
        print("ERROR: Could not load models")
        return

    n = fid.load_db()
    if n == 0:
        print("No enrolled identities. Enroll someone first.")
        return

    print(f"\nTesting identification ({n} enrolled identities)...")
    frame = fetch_frame()
    if frame is None:
        print("Could not fetch frame from streamerpi")
        return

    t0 = time.time()
    results = fid.identify(frame)
    dt = (time.time() - t0) * 1000

    if not results:
        print(f"No faces detected ({dt:.0f}ms)")
        return

    print(f"\nDetected {len(results)} face(s) in {dt:.0f}ms:")
    for r in results:
        print(f"  - {r['name']} (distance={r['distance']:.3f}, "
              f"confidence={r['confidence']}, bbox={r['bbox']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face enrollment for Little Timmy")
    parser.add_argument("name", nargs="?", help="Name to enroll")
    parser.add_argument("--count", type=int, default=15, help="Number of frames to capture")
    parser.add_argument("--interval", type=float, default=0.7, help="Seconds between captures")
    parser.add_argument("--list", action="store_true", help="List enrolled identities")
    parser.add_argument("--delete", type=str, help="Delete an enrolled identity")
    parser.add_argument("--push", action="store_true", help="Push face DB to streamerpi")
    parser.add_argument("--test", action="store_true", help="Test identification on live frame")

    args = parser.parse_args()

    if args.push:
        push_to_pi()
    elif args.list:
        list_enrolled()
    elif args.delete:
        delete_enrolled(args.delete)
    elif args.test:
        test_identify()
    elif args.name:
        enroll(args.name, count=args.count, interval=args.interval)
    else:
        parser.print_help()
