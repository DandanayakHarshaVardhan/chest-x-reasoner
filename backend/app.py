import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io

from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ===== DenseNet Vision Model =====
VISION_WEIGHTS_PATH = "path\to\weight\.pth"

vision_model = models.densenet121(pretrained=False)
vision_model.classifier = nn.Linear(
    vision_model.classifier.in_features, 7
)

checkpoint = torch.load(VISION_WEIGHTS_PATH, map_location=DEVICE)
vision_model.load_state_dict(checkpoint["model_state"])
vision_model = vision_model.to(DEVICE)
vision_model.eval()

print("✅ DenseNet loaded")

LABELS = [
    "lung_opacity", 
    "consolidation",
    "pleural_effusion",
    "cardiomegaly",
    "atelectasis",
    "edema",
    "support_devices"
]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_findings_pil(img, threshold=0.5):
    img = image_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = vision_model(img)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    preds = {k: float(v) for k, v in zip(LABELS, probs)}
    present = [k.replace("_", " ") for k, v in preds.items() if v >= threshold]
    absent  = [k.replace("_", " ") for k, v in preds.items() if v < threshold]

    return preds, present, absent

# ===== Phi-3 Reasoning Model =====
PHI_MODEL = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL)
phi_model = AutoModelForCausalLM.from_pretrained(
    PHI_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)
phi_model.eval()

print("✅ Phi-3 loaded")


def build_reasoning_prompt(present, absent, question):
    system_prompt = (
        "You are a radiology reasoning assistant.\n"
        "give step by step resoning for the findings in xray in 4 to 5 lines, reason strictly from the findings\n"
        "You MUST output ONLY in the following format:\n\n"
        "Reasoning:\n"
        "provide step-by-step medical reasoning.\n"
        "Conclusion:\n"
        "<one small single sentence>\n\n"

    )

    user_prompt = f"""
Detected findings:
Present: {', '.join(present) if present else 'None'}
Absent: {', '.join(absent) if absent else 'None'}

Question:
{question}
"""

    return f"<|system|>{system_prompt}<|user|>{user_prompt}<|assistant|>"



import re

def reason_with_phi(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(phi_model.device)

    with torch.no_grad():
        output_ids = phi_model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            temperature=0.0,
            top_p=1.0
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove assistant tag if leaked
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]

    # 🔥 Capture ALL Reasoning + Conclusion blocks
    matches = re.findall(
        r"Reasoning:\s*(.*?)\s*Conclusion:\s*(.*?)(?=Reasoning:|$)",
        text,
        re.S
    )

    if not matches:
        return (
            "Reasoning:\nUnable to generate reasoning.\n\n"
            "Conclusion:\nUnable to conclude."
        )

    # ✅ TAKE ONLY THE LAST (REAL) ONE
    reasoning, conclusion = matches[-1]

    return (
        "Reasoning:\n"
        + reasoning.strip()
        + "\n\nConclusion:\n"
        + conclusion.strip()
    )


from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import traceback

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    print("📥 Request received")

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        question = request.form.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        img = Image.open(io.BytesIO(image_file.read())).convert("RGB")

        _, present, absent = extract_findings_pil(img)

        prompt = build_reasoning_prompt(present, absent, question)
        output = reason_with_phi(prompt)

        return jsonify({
            "output": output
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


from pyngrok import ngrok
import threading

ngrok.set_auth_token("Ngrok_Authtoken")

public_url = ngrok.connect(5000)
print("🌍 PUBLIC URL:", public_url)

threading.Thread(
    target=app.run,
    kwargs={"host": "0.0.0.0", "port": 5000},
    daemon=True
).start()


