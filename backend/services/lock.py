"""
Single shared inference lock.

The GPU (or CPU, in dev) here has 4 GB of VRAM and cannot take concurrent
model inference. Every call into yolo_service.predict / unet_service.segment
must hold this semaphore for its duration. Deliberately a single shared
semaphore (not one per service) so YOLO and U-Net calls for the same request
serialize against each other too.
"""
import threading

INFERENCE_LOCK = threading.Semaphore(1)
