"""U-Net shape-validating loader: must name the exact mismatched tensor and
both shapes on failure, never load a mismatched head silently, and never
crash when no checkpoint is present."""
import torch

from backend.config import settings
from backend.ml.unet_arch import UNet
from backend.services.unet_service import UnetService, check_final_layer_shape


def test_matching_shapes_pass():
    model = UNet(in_channels=1, num_classes=3)
    state_dict = model.state_dict()
    assert check_final_layer_shape(state_dict, model) is None


def test_mismatched_out_channels_named_precisely():
    configured_model = UNet(in_channels=1, num_classes=3)
    checkpoint_model = UNet(in_channels=1, num_classes=5)  # e.g. checkpoint trained with 5 classes
    checkpoint_state = checkpoint_model.state_dict()

    message = check_final_layer_shape(checkpoint_state, configured_model)
    assert message is not None
    assert "final_conv.weight" in message
    # both shapes must be named
    assert "(5, 16, 1, 1)" in message or "5, 16, 1, 1" in message
    assert "(3, 16, 1, 1)" in message or "3, 16, 1, 1" in message


def test_missing_tensor_named():
    configured_model = UNet(in_channels=1, num_classes=3)
    incomplete_state = {k: v for k, v in configured_model.state_dict().items() if k != "final_conv.weight"}
    message = check_final_layer_shape(incomplete_state, configured_model)
    assert message is not None
    assert "final_conv.weight" in message


def test_no_checkpoint_reports_unavailable_not_crash():
    service = UnetService()
    status = service.status()
    assert status["loaded"] is False
    assert "weights not found" in status["reason"]


def test_shape_mismatch_checkpoint_on_disk_reports_precise_reason(tmp_path, monkeypatch):
    # Write a real checkpoint with the wrong number of output classes.
    wrong_model = UNet(in_channels=1, num_classes=7)
    ckpt_path = tmp_path / "unet_wrong.pth"
    torch.save(wrong_model.state_dict(), ckpt_path)

    monkeypatch.setattr(settings, "unet_model_path", str(ckpt_path))
    monkeypatch.setattr(settings, "unet_num_classes", 3)

    service = UnetService()
    status = service.status()
    assert status["loaded"] is False
    assert "shape mismatch" in status["reason"]
    assert "final_conv.weight" in status["reason"]
    assert "UNET_NUM_CLASSES=3" in status["reason"]
