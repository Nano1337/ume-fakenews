import gradio as gr
from PIL import Image
import torch
from transformers import AutoProcessor
from fakeddit.get_data import FakedditDataset
from fakeddit import get_model
import argparse
from utils.setup_configs import setup_configs

# Argument parser setup
parser = argparse.ArgumentParser(description="Specify the directory and model parameters")
parser.add_argument("--dir", type=str, help="directory to run")
parser.add_argument("--model_type", type=str, help="model type to run")
parser.add_argument("--ckpt", type=str, help="checkpoint to load")
args = parser.parse_args()
args = setup_configs(parser=parser)

# Set device for model execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model(args)
checkpoint = torch.load(args.ckpt, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)

# Load processor
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

def classify_image_text(image, text):
    # Preprocess inputs
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True)
    text_tokens = inputs["input_ids"]
    img_tokens = inputs["pixel_values"]
    label = torch.tensor([1]).to(device)  # Dummy label

    # Move inputs to device
    text_tokens = text_tokens.to(device)
    img_tokens = img_tokens.to(device)

    # Model inference
    with torch.no_grad():
        x1_logits, x2_logits, _, _ = model(text_tokens, img_tokens, label)
        logits = (x1_logits + x2_logits) / 2
    
    # Prediction
    prediction = logits.argmax(-1).item()
    result = "real" if prediction == 1 else "fake"
    return result

# Example inputs
examples = [
    ["pics/demo_images/6zgaac.webp", "Reporter Rescues Two Dolphins While Covering Hurricane Irma"],
    ["pics/demo_images/b1d1hj.png", "North Korean Air Force bombing Seoul during the Korean War (1950)"],
    ["pics/demo_images/bg38c7.webp", "The clouds blocked the Empire State Building from view outside my window so that it looked like it disappeared."],
    ["pics/demo_images/comv5l9.jpeg", "Obama really is a lizard person"],
]

# Define Gradio interface
demo = gr.Interface(fn=classify_image_text, inputs=['image', 'text'], outputs='text',
                    title="Fake or Real Classifier", examples=examples)

# Launch the demo
demo.launch()
