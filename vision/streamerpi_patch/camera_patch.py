"""Patch for streamerpi camera.py — safe frame buffer for /capture endpoint.

Problem: async_get_capture_jpeg pulled from video_player.video.recv() directly,
which caused unbounded memory growth when no WebRTC client was consuming frames.

Fix: Add a background task (frame_buffer_task) that periodically reads one frame
from the video_player and stores it in latest_video_frame. The /capture endpoint
then just reads from this buffer (sync get_capture_jpeg), no async needed.

Apply this patch by:
1. Adding start_frame_buffer() / stop_frame_buffer() to camera.py
2. Calling start_frame_buffer() in server.py on_startup
3. Removing async_get_capture_jpeg entirely
4. Reverting server.py capture_frame handler to use sync get_capture_jpeg()
"""

# === ADD to camera.py after the existing globals (around line 65) ===

# _frame_buffer_task: asyncio.Task | None = None

# === ADD these functions to camera.py ===

FRAME_BUFFER_CODE = '''
_frame_buffer_task = None

async def _frame_buffer_loop():
    """Background task: read one frame per second from video_player into the buffer.

    This keeps latest_video_frame populated even when no WebRTC client is connected.
    Only reads when the buffer is empty or stale (>2s old), to avoid unnecessary work
    when a WebRTC client is already feeding the buffer via FaceDetectionVideoTransformer.
    """
    global latest_video_frame, last_frame_time
    while True:
        try:
            await asyncio.sleep(1.0)
            # Skip if buffer is fresh (WebRTC client is feeding it)
            if latest_video_frame is not None and (time.time() - last_frame_time) < 2.0:
                continue
            # Pull one frame from video_player
            if video_player and video_player.video:
                frame = await video_player.video.recv()
                with frame_lock:
                    latest_video_frame = frame
                    last_frame_time = time.time()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug("Frame buffer loop error: %s", e)
            await asyncio.sleep(2.0)


def start_frame_buffer(loop):
    """Start the background frame buffer task."""
    global _frame_buffer_task
    _frame_buffer_task = loop.create_task(_frame_buffer_loop())
    logger.info("Frame buffer started (1 fps background capture)")


def stop_frame_buffer():
    """Stop the background frame buffer task."""
    global _frame_buffer_task
    if _frame_buffer_task:
        _frame_buffer_task.cancel()
        _frame_buffer_task = None
'''

# === CHANGES to server.py ===

SERVER_IMPORT_CHANGE = """
# In the 'from camera import ...' line, ADD: start_frame_buffer, stop_frame_buffer
# REMOVE: async_get_capture_jpeg

# In on_startup, after start_camera(), ADD:
#     start_frame_buffer(asyncio.get_event_loop())

# In on_shutdown, before stop_camera(), ADD:
#     stop_frame_buffer()

# In capture_frame handler, REVERT to:
#     jpeg_bytes, age = get_capture_jpeg()
"""
