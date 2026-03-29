"""Run this ON streamerpi to apply the camera fix.

Usage: python3 apply_patch.py
"""
import re

CAM_PATH = "/home/pi/little_timmy_motor_raspi/camera.py"
SRV_PATH = "/home/pi/little_timmy_motor_raspi/server.py"

# --- Patch camera.py ---
with open(CAM_PATH, "r") as f:
    cam = f.read()

# 1. Remove async_get_capture_jpeg entirely
if "async def async_get_capture_jpeg" in cam:
    # Find and remove the async function
    start = cam.index("async def async_get_capture_jpeg")
    # Find next function or end
    next_def = cam.find("\ndef ", start + 1)
    if next_def == -1:
        cam = cam[:start].rstrip() + "\n"
    else:
        cam = cam[:start] + cam[next_def:]
    print("  Removed async_get_capture_jpeg")

# 2. Restore simple get_capture_jpeg (in case it was modified)
if "# Pull a fresh frame from video_player" in cam:
    # It has the fallback code, simplify it
    old_start = cam.index("def get_capture_jpeg")
    old_end = cam.index("\n\n", old_start + 50)
    simple_func = '''def get_capture_jpeg(quality=85):
    """Grab the latest buffered video frame as JPEG bytes.

    Returns (jpeg_bytes, frame_age_seconds) or (None, None) if unavailable.
    """
    with frame_lock:
        if latest_video_frame is None:
            return None, None
        age = time.time() - last_frame_time
        try:
            img = latest_video_frame.to_ndarray(format="bgr24")
        except Exception:
            return None, None

    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes(), age'''
    cam = cam[:old_start] + simple_func + cam[old_end:]
    print("  Simplified get_capture_jpeg")

# 3. Add import asyncio if not present
if "import asyncio" not in cam:
    cam = cam.replace("import threading", "import asyncio\nimport threading")
    print("  Added asyncio import")

# 4. Add frame buffer code before get_capture_jpeg
frame_buffer_code = '''
_frame_buffer_task = None


async def _frame_buffer_loop():
    """Background: read one frame/sec from video_player into the buffer.

    Keeps latest_video_frame populated without a WebRTC client.
    Skips when buffer is fresh (<2s old) to avoid double-reading.
    """
    global latest_video_frame, last_frame_time
    while True:
        try:
            await asyncio.sleep(1.0)
            if latest_video_frame is not None and (time.time() - last_frame_time) < 2.0:
                continue
            if video_player and video_player.video:
                frame = await video_player.video.recv()
                with frame_lock:
                    latest_video_frame = frame
                    last_frame_time = time.time()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug("Frame buffer error: %s", e)
            await asyncio.sleep(2.0)


def start_frame_buffer(loop):
    """Start background frame buffer (1fps capture when no WebRTC client)."""
    global _frame_buffer_task
    _frame_buffer_task = loop.create_task(_frame_buffer_loop())
    logger.info("Frame buffer started")


def stop_frame_buffer():
    """Stop background frame buffer."""
    global _frame_buffer_task
    if _frame_buffer_task:
        _frame_buffer_task.cancel()
        _frame_buffer_task = None


'''

if "_frame_buffer_task" not in cam:
    # Insert before get_capture_jpeg
    insert_point = cam.index("def get_capture_jpeg")
    cam = cam[:insert_point] + frame_buffer_code + cam[insert_point:]
    print("  Added frame buffer code")

with open(CAM_PATH, "w") as f:
    f.write(cam)
print("camera.py patched")

# --- Patch server.py ---
with open(SRV_PATH, "r") as f:
    srv = f.read()

# 1. Fix import — remove async_get_capture_jpeg, add start/stop_frame_buffer
srv = srv.replace(
    "async_get_capture_jpeg",
    "start_frame_buffer, stop_frame_buffer"
)
# Deduplicate if get_capture_jpeg is listed twice
srv = srv.replace(
    "get_capture_jpeg, start_frame_buffer, stop_frame_buffer",
    "get_capture_jpeg, start_frame_buffer, stop_frame_buffer"
)
print("  Fixed imports")

# 2. Revert capture handler to sync
srv = srv.replace(
    "jpeg_bytes, age = await async_get_capture_jpeg()",
    "jpeg_bytes, age = get_capture_jpeg()"
)
print("  Reverted capture handler to sync")

# 3. Add start_frame_buffer to on_startup (after start_camera())
if "start_frame_buffer" not in srv.split("on_startup")[1] if "on_startup" in srv else "":
    srv = srv.replace(
        "    start_camera()\n",
        "    start_camera()\n    start_frame_buffer(asyncio.get_event_loop())\n"
    )
    print("  Added start_frame_buffer to on_startup")

# 4. Add stop_frame_buffer to on_shutdown (before stop_camera())
if "stop_frame_buffer" not in srv.split("on_shutdown")[1] if "on_shutdown" in srv else "":
    srv = srv.replace(
        "    stop_face_tracking(state)\n    stop_camera()",
        "    stop_face_tracking(state)\n    stop_frame_buffer()\n    stop_camera()"
    )
    print("  Added stop_frame_buffer to on_shutdown")

with open(SRV_PATH, "w") as f:
    f.write(srv)
print("server.py patched")

# Verify syntax
import ast
ast.parse(open(CAM_PATH).read())
print("camera.py syntax OK")
ast.parse(open(SRV_PATH).read())
print("server.py syntax OK")
print("\nAll patches applied successfully!")
