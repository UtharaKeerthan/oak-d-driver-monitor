"""
download_models.py
Downloads pre-compiled OpenVINO .blob models for the OAK-D VPU.
"""
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

def download_models():
    MODELS_DIR.mkdir(exist_ok=True)
    try:
        import blobconverter
        print("Downloading face-detection-adas-0001...")
        blobconverter.from_zoo(
            name="face-detection-adas-0001",
            shaves=6,
            output_dir=str(MODELS_DIR),
        )
        print("Downloading head-pose-estimation-adas-0001...")
        blobconverter.from_zoo(
            name="head-pose-estimation-adas-0001",
            shaves=4,
            output_dir=str(MODELS_DIR),
        )
        print("[OK] Models downloaded to", MODELS_DIR)
    except ImportError:
        print("Install blobconverter: pip install blobconverter")

if __name__ == "__main__":
    download_models()
