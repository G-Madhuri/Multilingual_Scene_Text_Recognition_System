import zipfile
import os
import sys
import warnings
import logging
import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# =========================
# Auto-unzip parseq.zip if it exists
# =========================
if os.path.exists('parseq.zip'):
    print("Found parseq.zip, extracting...")
    try:
        with zipfile.ZipFile('parseq.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove('parseq.zip')
        print("✅ Extracted and removed parseq.zip")
    except Exception as e:
        print(f"Error extracting parseq.zip: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# Setup PARSeq path - USE LOCAL FOLDER, NOT TORCH HUB
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
parseq_local_path = os.path.join(current_dir, 'parseq')

if os.path.exists(parseq_local_path):
    sys.path.insert(0, parseq_local_path)
    logger.info(f"✅ Using local parseq folder at {parseq_local_path}")
else:
    logger.error(f"parseq folder not found at {parseq_local_path}")
    sys.exit(1)

# Import from local parseq folder
try:
    from strhub.data.utils import Tokenizer
    from strhub.models.parseq.model import PARSeq
    logger.info("✅ Successfully imported Tokenizer and PARSeq from local folder")
except ImportError as e:
    logger.error(f"Failed to import: {e}")
    logger.info(f"Contents of parseq folder: {os.listdir(parseq_local_path)}")
    raise

warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
ORIYA_CHARSET = "ଅଆଇଈଉଊଋଌଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହାିିୀୁୂୃୄେୈୋୌ୍ଂଁଃ"

LANGUAGES = {
    "Telugu": {
        "model_path": "parseq_telugu_finetuned_final_5epochs.pth",
        "samples_dir": "telugu_samples",
    },
    "Bengali": {
        "model_path": "finetuned_bengali_model.pth",
        "samples_dir": "bengali_samples",
    },
    "Oriya": {
        "model_path": "parseq_oriya_final_direct.pth",
        "samples_dir": "oriya_samples",
        "charset": ORIYA_CHARSET,
    }
}

# =========================
# Image Transform
# =========================
transform = T.Compose([
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# =========================
# Decode
# =========================
def decode_prediction(logits, tokenizer):
    pred_ids = logits.argmax(-1)[0]
    chars = []
    for t in pred_ids:
        t = t.item()
        if t == tokenizer.eos_id:
            break
        if t not in [tokenizer.pad_id, tokenizer.bos_id] and t < len(tokenizer._itos):
            chars.append(tokenizer._itos[t])
    return "".join(chars)

# =========================
# Model Cache
# =========================
model_cache = {}

def load_model(model_path, lang_name):
    cache_key = f"{lang_name}_{model_path}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading {lang_name} model on {device}")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None, None, None

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        logger.info(f"Checkpoint loaded for {lang_name}")

        if 'charset' in checkpoint:
            charset_str = checkpoint['charset']
        elif lang_name == "Oriya":
            charset_str = ORIYA_CHARSET
        else:
            logger.warning(f"No charset found for {lang_name}")
            return None, None, None

        logger.info(f"Charset length for {lang_name}: {len(charset_str)}")

        # Create tokenizer
        tokenizer = Tokenizer(charset_str)
        
        # Create model instance from local PARSeq class
        # Get model parameters from state dict keys
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        # Create model with default parameters (it will be overwritten by state_dict)
        model = PARSeq()
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v
        
        # Load weights
        model.load_state_dict(new_state_dict, strict=False)
        model.tokenizer = tokenizer
        model = model.to(device)
        model.eval()

        model_cache[cache_key] = (model, device, tokenizer)
        logger.info(f"✅ Loaded {lang_name} model successfully")
        return model, device, tokenizer

    except Exception as e:
        logger.error(f"Error loading {lang_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# =========================
# Inference
# =========================
def inference_image(model, image, device, tokenizer):
    if image is None:
        return "", 0.0
    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        predicted_text = decode_prediction(logits, tokenizer)

        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1)[0][0]
        avg_conf = max_probs[:len(predicted_text)].mean().item() if len(predicted_text) > 0 else 0

    return predicted_text, avg_conf

# =========================
# Get samples for specific language
# =========================
def get_samples_for_language(language):
    """Get sample images for a specific language"""
    config = LANGUAGES[language]
    folder = config["samples_dir"]
    samples = []
    
    if os.path.exists(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append(os.path.join(folder, f))
    
    return samples[:6]

# =========================
# Create a tab for each language
# =========================
def create_language_tab(language):
    """Create a tab interface for a specific language"""
    
    # Get samples for this language
    sample_images = get_samples_for_language(language)
    
    with gr.Row():
        # Left column - Image preview and controls
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", 
                label=f"📷 {language} Image Preview",
                height=350,
                interactive=True
            )
            
            # Extract button right below the preview
            extract_btn = gr.Button(
                f"✨ Extract Text", 
                variant="primary"
            )
            
            # Sample images section
            if sample_images:
                gr.Markdown("---")
                gr.Markdown(f"### 📸 Click any {language} sample image to preview")
                
                # Create gallery that doesn't expand when clicked
                sample_gallery = gr.Gallery(
                    value=sample_images,
                    label=f"{language} Sample Images",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    allow_preview=False,
                    interactive=False
                )
                
                # Function to update main preview when sample is selected
                def update_preview_from_sample(evt: gr.SelectData):
                    selected_index = evt.index
                    selected_image_path = sample_images[selected_index]
                    return Image.open(selected_image_path)
                
                sample_gallery.select(
                    update_preview_from_sample,
                    outputs=image_input
                )
        
        # Right column - Results
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="📝 Extracted Text",
                lines=6,
                placeholder="Extracted text will appear here...",
                interactive=False
            )
            confidence = gr.Textbox(
                label="🎯 Confidence Score",
                placeholder="Confidence will appear here...",
                interactive=False
            )
    
    # Handle prediction
    def predict_wrapper(image):
        if image is None:
            return "⚠️ Please upload or select an image first", ""
        
        model_path = LANGUAGES[language]["model_path"]
        model, device, tokenizer = load_model(model_path, language)
        
        if model is None:
            return f"❌ Failed to load {language} model. Please check if the model file exists and is valid.", ""
        
        text, conf = inference_image(model, image, device, tokenizer)
        
        if text == "":
            return "🔍 No text detected in the image", ""
        
        return text, f"✅ Confidence: {conf:.2%}"
    
    extract_btn.click(
        fn=predict_wrapper,
        inputs=[image_input],
        outputs=[output_text, confidence]
    )
    
    return image_input

# =========================
# Main UI with Tabs
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Multilingual Scene Text Recognition", css="""
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
    }
    
    .tab-nav button {
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 12px 24px !important;
        color: #000000 !important;
        background-color: #f0f0f0 !important;
        border: 2px solid #ccc !important;
        margin-right: 8px !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    .tab-nav button.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .tab-nav button:hover {
        background-color: #e0e0e0 !important;
        transform: translateY(-2px);
    }
    
    button {
        transition: all 0.3s ease !important;
        font-weight: bold !important;
        font-size: 16px !important;
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    
    .gr-gallery {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background-color: #fafafa;
    }
    
    .gr-gallery .gallery-item {
        cursor: pointer !important;
        transition: transform 0.2s !important;
    }
    
    .gr-gallery .gallery-item:hover {
        transform: scale(1.05) !important;
    }
    
    .gr-box {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
""") as demo:
    
    gr.Markdown("""
    # 📖 Multilingual Scene Text Recognition System
    ### Extract text from images in Telugu, Bengali, and Oriya languages
    
    ---
    """)
    
    # Create tabs for each language
    with gr.Tabs():
        for lang in LANGUAGES.keys():
            with gr.TabItem(f"🔤 {lang}"):
                create_language_tab(lang)
    
    gr.Markdown("""
    ---
    ### 💡 How to use:
    1. **Select a language tab** (Telugu, Bengali, or Oriya)
    2. **Click any sample thumbnail** - it will load into the main preview above
    3. **Click "Extract Text"** button below the preview
    4. **View results** on the right side
    """)

# =========================
# Run
# =========================
if __name__ == "__main__":
    for lang, config in LANGUAGES.items():
        if not os.path.exists(config["model_path"]):
            logger.warning(f"⚠️ Model not found: {config['model_path']} for {lang}")
        if not os.path.exists(config["samples_dir"]):
            os.makedirs(config["samples_dir"], exist_ok=True)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )