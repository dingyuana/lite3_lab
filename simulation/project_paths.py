from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LITE3_MODEL_ROOT = (
    PROJECT_ROOT
    / "Lite3_rl_deploy"
    / "third_party"
    / "deep_robotics_model"
    / "Lite3"
)
LITE3_XML = LITE3_MODEL_ROOT / "Lite3_mjcf" / "mjcf" / "Lite3.xml"
LITE3_URDF = LITE3_MODEL_ROOT / "Lite3_urdf" / "urdf" / "Lite3.urdf"
POLICY_ONNX = PROJECT_ROOT / "Lite3_rl_deploy" / "policy" / "ppo" / "policy.onnx"
