import io
import os
from pathlib import Path

# Helps avoid native BLAS/OpenMP thread crashes on some macOS setups.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
CNN_MODEL_PATH = PROJECT_ROOT.parent / "models" / "brain_mri_densenet121_best.pt"
VLM_MODEL_NAME = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
IMG_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


app = Flask(__name__)


# Loaded once at startup
CNN_DEVICE = None
CNN_MODEL = None
CLASS_NAMES = None

VLM_MODEL = None
VLM_PROCESSOR = None
VLM_INPUT_DTYPE = None
VLM_DEVICE_NAME = None
MODELS_READY = False
MODEL_INIT_ERROR = None

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_env_file(env_path):
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


load_env_file(REPO_ROOT / ".env")
load_env_file(PROJECT_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN")


def resolve_runtime_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


RUNTIME_DEVICE = resolve_runtime_device()
if RUNTIME_DEVICE == "mps":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device():
    return torch.device(RUNTIME_DEVICE)


def build_densenet121(num_classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def load_cnn_checkpoint(ckpt_path, device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_densenet121(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_names


def get_model_device(model):
    return next(model.parameters()).device


def load_vlm():
    global VLM_DEVICE_NAME
    model_kwargs = {}
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN

    if RUNTIME_DEVICE in {"cuda", "mps"}:
        dtype = torch.float16
        model = LlavaForConditionalGeneration.from_pretrained(
            VLM_MODEL_NAME,
            torch_dtype=dtype,
            **model_kwargs,
        )
    else:
        dtype = torch.float32
        model = LlavaForConditionalGeneration.from_pretrained(
            VLM_MODEL_NAME,
            **model_kwargs,
        )

    vlm_device = get_device()
    model.to(vlm_device)
    VLM_DEVICE_NAME = str(vlm_device)

    print(f"[predict] Using", VLM_DEVICE_NAME)
    processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME, **model_kwargs)
    model.eval()
    return model, processor, dtype


def build_explanation_prompt(predicted_label):
    question = (
        f"This brain MRI suggests {predicted_label}. "
        "What findings in the image support this diagnosis? "
    )
    return f"USER: <image>\n{question} ASSISTANT:"


def generate_explanation(image, predicted_label):
    if VLM_MODEL is None or VLM_PROCESSOR is None or VLM_INPUT_DTYPE is None:
        raise RuntimeError("VLM is not loaded.")

    prompt = build_explanation_prompt(predicted_label)
    inputs = VLM_PROCESSOR(images=[image], text=prompt, return_tensors="pt")

    model_device = get_model_device(VLM_MODEL)
    casted_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(model_device)
            if v.is_floating_point():
                v = v.to(VLM_INPUT_DTYPE)
        casted_inputs[k] = v

    try:
        with torch.inference_mode():
            generated_ids = VLM_MODEL.generate(**casted_inputs, max_new_tokens=220)
    except ValueError as exc:
        # Match prior project behavior for occasional image/token alignment failures.
        if "Image features and image tokens do not match" not in str(exc):
            raise
        fallback_prompt = f"<image>\n{prompt}"
        retry_inputs = VLM_PROCESSOR(images=[image], text=fallback_prompt, return_tensors="pt")
        casted_retry_inputs = {}
        for k, v in retry_inputs.items():
            if torch.is_tensor(v):
                v = v.to(model_device)
                if v.is_floating_point():
                    v = v.to(VLM_INPUT_DTYPE)
            casted_retry_inputs[k] = v

        with torch.inference_mode():
            generated_ids = VLM_MODEL.generate(**casted_retry_inputs, max_new_tokens=220)
        casted_inputs = casted_retry_inputs

    input_token_count = casted_inputs["input_ids"].shape[-1]
    generated_only = generated_ids[0][input_token_count:]
    explanation = VLM_PROCESSOR.decode(generated_only, skip_special_tokens=True).strip()
    return explanation


def initialize_models():
    global CNN_DEVICE, CNN_MODEL, CLASS_NAMES, VLM_MODEL, VLM_PROCESSOR, VLM_INPUT_DTYPE
    global MODELS_READY, MODEL_INIT_ERROR

    if MODELS_READY:
        return

    try:
        print("[startup] MPS available:", torch.backends.mps.is_available())
        print("[startup] CUDA available:", torch.cuda.is_available())
        CNN_DEVICE = get_device()
        print("[startup] Selected CNN device:", CNN_DEVICE)
        print("[startup] Loading CNN checkpoint...")
        CNN_MODEL, CLASS_NAMES = load_cnn_checkpoint(CNN_MODEL_PATH, CNN_DEVICE)
        print("[startup] CNN loaded.")
        print("[startup] Loading VLM model...")
        VLM_MODEL, VLM_PROCESSOR, VLM_INPUT_DTYPE = load_vlm()
        print("[startup] VLM loaded on:", VLM_DEVICE_NAME)
        MODELS_READY = True
        MODEL_INIT_ERROR = None
        print("[startup] All models ready.")
    except Exception as exc:
        MODEL_INIT_ERROR = str(exc)
        MODELS_READY = False
        print("[startup] Model initialization failed:", MODEL_INIT_ERROR)


def extract_uploaded_file_bytes():
    file_obj = request.files.get("file")
    if file_obj is not None:
        return file_obj.read()

    raw_body = request.get_data() or b""
    if not raw_body:
        return None

    content_type = (request.headers.get("Content-Type") or "").lower()
    if "multipart/form-data" not in content_type:
        return None

    # Fallback for clients that send multipart/form-data without boundary header.
    first_line = raw_body.split(b"\r\n", 1)[0]
    if not first_line.startswith(b"--"):
        first_line = raw_body.split(b"\n", 1)[0]
    if not first_line.startswith(b"--"):
        return None

    boundary = first_line[2:].strip()
    if not boundary:
        return None

    marker = b"--" + boundary
    for part in raw_body.split(marker):
        if b'name="file"' not in part:
            continue

        if b"\r\n\r\n" in part:
            _, file_bytes = part.split(b"\r\n\r\n", 1)
        elif b"\n\n" in part:
            _, file_bytes = part.split(b"\n\n", 1)
        else:
            continue

        file_bytes = file_bytes.rstrip(b"\r\n")
        if file_bytes.endswith(b"--"):
            file_bytes = file_bytes[:-2]
        file_bytes = file_bytes.rstrip(b"\r\n")
        if file_bytes:
            return file_bytes

    return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "models_ready": MODELS_READY,
            "model_init_error": MODEL_INIT_ERROR,
            "device": {
                "selected_device": str(get_device()),
                "cnn_device": str(CNN_DEVICE) if CNN_DEVICE is not None else None,
                "vlm_device": VLM_DEVICE_NAME,
                "mps_available": torch.backends.mps.is_available(),
                "cuda_available": torch.cuda.is_available(),
            },
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    print("[predict] request received", flush=True)

    if not MODELS_READY:
        print("[predict] models not ready, initializing...", flush=True)
        initialize_models()
        if not MODELS_READY:
            print("[predict] model initialization failed", flush=True)
            return jsonify({"detail": f"Model initialization failed: {MODEL_INIT_ERROR}"}), 500

    if CNN_MODEL is None or CLASS_NAMES is None:
        print("[predict] CNN model missing", flush=True)
        return jsonify({"detail": "CNN model not loaded."}), 500
    if VLM_MODEL is None or VLM_PROCESSOR is None:
        print("[predict] VLM model missing", flush=True)
        return jsonify({"detail": "VLM not loaded."}), 500

    print("[predict] reading multipart payload...", flush=True)
    content = extract_uploaded_file_bytes()
    if content is None:
        print("[predict] no file found in request", flush=True)
        return jsonify({"detail": "Missing file field in form-data."}), 400

    try:
        print(f"[predict] received file bytes: {len(content)}", flush=True)
        image = Image.open(io.BytesIO(content)).convert("RGB")
        print(f"[predict] image decoded size={image.size}", flush=True)
    except (UnidentifiedImageError, OSError):
        print("[predict] invalid image payload", flush=True)
        return jsonify({"detail": "Invalid image file."}), 400

    x = EVAL_TRANSFORM(image).unsqueeze(0).to(CNN_DEVICE)
    print(f"[predict] running CNN inference on device={CNN_DEVICE}", flush=True)

    with torch.inference_mode():
        logits = CNN_MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        predicted_label = CLASS_NAMES[pred_idx]
    print(
        f"[predict] cnn result label={predicted_label} confidence={confidence:.4f}",
        flush=True,
    )

    try:
        print("[predict] generating VLM explanation...", flush=True)
        explanation = generate_explanation(image, predicted_label)
        print("[predict] explanation generated", flush=True)
    except Exception as exc:
        print(f"[predict] explanation failed: {exc}", flush=True)
        return jsonify({"detail": f"Explanation generation failed: {exc}"}), 500

    print("[predict] returning response", flush=True)
    return jsonify(
        {
            "prediction": predicted_label,
            "confidence": confidence,
            "explanation": explanation,
        }
    )


if __name__ == "__main__":
    initialize_models()
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=False, use_reloader=False)
