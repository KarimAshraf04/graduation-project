"""Quick check that PyTorch (YOLO) and onnxruntime (faces) can both use the GPU."""
import torch

print("=" * 50)
ok_torch = torch.cuda.is_available()
print(f"PyTorch {torch.__version__}  CUDA available: {ok_torch}")
if ok_torch:
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

try:
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:
            print(f"  preload_dlls warning: {e}")
    providers = ort.get_available_providers()
    ok_ort = "CUDAExecutionProvider" in providers
    print(f"onnxruntime {ort.__version__}  providers: {providers}")
except Exception as e:
    ok_ort = False
    print(f"onnxruntime not working: {e}")

print("=" * 50)
print("YOLO  ->", "GPU  OK" if ok_torch else "CPU  (slow!)")
print("Faces ->", "GPU  OK" if ok_ort else "CPU  (slow!)")
if not ok_torch:
    print("Fix: venv\\Scripts\\python -m pip install torch torchvision "
          "--index-url https://download.pytorch.org/whl/cu128 --force-reinstall")
if not ok_ort:
    print("Fix: venv\\Scripts\\python -m pip uninstall -y onnxruntime onnxruntime-gpu && "
          "venv\\Scripts\\python -m pip install onnxruntime-gpu==1.22.0")
