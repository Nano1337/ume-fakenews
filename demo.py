import gradio as gr
from PIL import Image
import torch
from transformers import AutoProcessor
from fakeddit.get_data import FakedditDataset
from fakeddit import get_model
import argparse
from utils.setup_configs import setup_configs

parser = argparse.ArgumentParser(description="which directory to run")
parser.add_argument("--dir", type=str, default=None, help="directory to run")
parser.add_argument("--model_type", type=str, default=None, help="model type to run")
parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to load")
args = parser.parse_args()
model_type = args.model_type
ckpt = args.ckpt
args = setup_configs(parser=parser)

setattr(args, "model_type", str(args.model_type))
setattr(args, "ckpt", str(args.ckpt))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load in model
model = get_model(args)
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

def classify_image_text(image, text):
    # Preprocess inputs
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True)
    text_tokens = inputs["input_ids"]
    img_tokens = inputs["pixel_values"]

    # Move inputs to device
    text_tokens = text_tokens.to(device)
    img_tokens = img_tokens.to(device)
    label = torch.tensor([0]).to(device) # ignore this
        

    # model inference
    with torch.no_grad():
        x1_logits, x2_logits, _, _ = model(text_tokens, img_tokens, label)
        logits = (x1_logits + x2_logits) / 2
    
    # convert to predicted class
    prediction = logits.argmax(-1).item()  # Convert to Python int
    
    # Map your model's output to "fake" or "real"
    result = "real" if prediction == 1 else "fake"
    return result

# Define Gradio interface
inputs = ['image', 'text']
outputs = 'text'

demo = gr.Interface(fn=classify_image_text, inputs=inputs, outputs=outputs, title="Fake or Real Classifier")

# Launch the demo
demo.launch()