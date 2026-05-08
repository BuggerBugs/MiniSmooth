import torch
import torchvision.models as models

# ---- SETTINGS ----
model_path = "/home/users/ntu/sooq0001/mydir/resnet18_imagenet.pth.tar"  # your checkpoint
onnx_path = "resnet18_fp32.onnx"
num_classes = 1000

# ---- LOAD MODEL ----
model = models.resnet18(num_classes=num_classes)

# Load checkpoint (handle state_dict inside checkpoint)
checkpoint = torch.load(model_path, map_location='cuda')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Remove 'module.' prefix if trained with DataParallel
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# Move to GPU and set to eval mode
model.eval().cuda()

# ---- DUMMY INPUT ----
dummy_input = torch.randn(1, 3, 224, 224).cuda()

# ---- EXPORT TO ONNX ----
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)

print(f"ONNX model exported to {onnx_path}")



