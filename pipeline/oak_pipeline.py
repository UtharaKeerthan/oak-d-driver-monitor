"""
oak_pipeline.py
DepthAI pipeline builder. Runs face detection + head pose on the OAK-D VPU.
Streams results to host for EAR/PERCLOS/yawn analysis.
"""
import depthai as dai
from pathlib import Path

FACE_DET_MODEL  = Path(__file__).parent / "models/face-detection-adas-0001.blob"
HEAD_POSE_MODEL = Path(__file__).parent / "models/head-pose-estimation-adas-0001.blob"

def build_pipeline(sil_mode: bool = False) -> dai.Pipeline:
    """
    Build the DepthAI node graph.
    In SIL_MODE the camera node is replaced by an XLinkIn feeding mock frames.
    """
    pipeline = dai.Pipeline()

    # --- Camera input ---
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(672, 384)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    # --- Face detection on VPU ---
    face_det = pipeline.create(dai.node.NeuralNetwork)
    face_det.setBlobPath(str(FACE_DET_MODEL))
    face_det.setNumInferenceThreads(2)
    cam.preview.link(face_det.input)

    # --- Image manipulator: crop face ROI ---
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(60, 60)
    manip.setMaxOutputFrameSize(60 * 60 * 3)

    # --- Head pose estimation on VPU ---
    head_pose = pipeline.create(dai.node.NeuralNetwork)
    head_pose.setBlobPath(str(HEAD_POSE_MODEL))
    head_pose.setNumInferenceThreads(2)
    manip.out.link(head_pose.input)

    # --- XLink outputs to host ---
    xout_preview   = pipeline.create(dai.node.XLinkOut)
    xout_face_det  = pipeline.create(dai.node.XLinkOut)
    xout_head_pose = pipeline.create(dai.node.XLinkOut)

    xout_preview.setStreamName("preview")
    xout_face_det.setStreamName("face_det")
    xout_head_pose.setStreamName("head_pose")

    cam.preview.link(xout_preview.input)
    face_det.out.link(xout_face_det.input)
    head_pose.out.link(xout_head_pose.input)

    return pipeline
