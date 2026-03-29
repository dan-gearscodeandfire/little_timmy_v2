"""Capture endpoint patch for streamerpi.

Deploy this to streamerpi's server.py to add a /capture endpoint
that returns a raw JPEG frame from the camera.

Usage: Add to server.py routes:
    app.router.add_get('/capture', capture_handler)

And import this handler, or inline the function.
"""

# --- Add this handler to streamerpi's server.py ---
# Assumes `camera` is the camera module with access to the OpenCV VideoCapture

import cv2
from aiohttp import web


async def capture_handler(request):
    """Return a single JPEG frame from the camera.

    GET /capture
    Returns: image/jpeg with raw JPEG bytes
    """
    try:
        # Access the global camera object
        # In streamerpi, this is typically camera.cap (cv2.VideoCapture)
        cap = request.app.get("camera_cap")
        if cap is None or not cap.isOpened():
            return web.Response(status=503, text="Camera not available")

        ret, frame = cap.read()
        if not ret or frame is None:
            return web.Response(status=503, text="Failed to capture frame")

        # Encode as JPEG (quality 85 for good size/quality tradeoff)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, jpeg_bytes = cv2.imencode('.jpg', frame, encode_params)
        if not success:
            return web.Response(status=500, text="JPEG encode failed")

        return web.Response(
            body=jpeg_bytes.tobytes(),
            content_type="image/jpeg",
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as e:
        return web.Response(status=500, text=f"Capture error: {e}")


# --- Integration instructions ---
# In streamerpi's server.py, add near the other route definitions:
#
#   from capture_endpoint import capture_handler
#   app.router.add_get('/capture', capture_handler)
#
# And in the app setup, store the camera capture object:
#
#   app['camera_cap'] = camera.cap  # or however the VideoCapture is accessed
#
# The camera.py module already has the VideoCapture. The exact reference
# depends on how camera.py exposes it. Check camera.py for the cap variable.
