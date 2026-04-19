import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys
import warnings
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# Setup PARSeq path
# =========================
parseq_path = os.path.join(os.path.dirname(__file__), 'parseq')
if os.path.exists(parseq_path):
    sys.path.insert(0, parseq_path)
else:
    logger.warning(f"PARSeq folder not found at {parseq_path}, trying direct import")

try:
    from strhub.data.utils import Tokenizer
except ImportError:
    logger.error("Failed to import Tokenizer. Make sure PARSeq is installed.")
    # Create a basic tokenizer if PARSeq is not available
    class Tokenizer:
        def __init__(self, charset):
            self.charset = charset
            self._itos = {i: ch for i, ch in enumerate(charset)}
            self._stoi = {ch: i for i, ch in enumerate(charset)}
            self.pad_id = 0
            self.bos_id = 1
            self.eos_id = 2

import torch.hub

warnings.filterwarnings('ignore')

# =========================
# Configuration - Added explicit charsets for all languages
# =========================
TELUGU_CHARSET = "అఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహాిీుూృౄెేైొోౌ్ౢౣ"
BENGALI_CHARSET = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহাািীুূৃৄেৈোৌ্ৎংঃ"
ORIYA_CHARSET = "ଅଆଇଈଉଊଋଌଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହାିିୀୁୂୃୄେୈୋୌ୍ଂଁଃ"

LANGUAGES = {
    "Telugu": {
        "model_path": "parseq_telugu_finetuned_final_5epochs.pth",
        "samples_dir": "telugu_samples",
        "charset": TELUGU_CHARSET
    },
    "Bengali": {
        "model_path": "finetuned_bengali_model.pth",
        "samples_dir": "bengali_samples",
        "charset": BENGALI_CHARSET
    },
    "Oriya": {
        "model_path": "parseq_oriya_final_direct.pth",
        "samples_dir": "oriya_samples",
        "charset": ORIYA_CHARSET
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
        # Load checkpoint with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        logger.info(f"Checkpoint loaded for {lang_name}")

        # Get charset - first try from checkpoint, then from config
        if 'charset' in checkpoint:
            charset_str = checkpoint['charset']
            logger.info(f"Using charset from checkpoint for {lang_name}")
        else:
            charset_str = LANGUAGES[lang_name].get('charset')
            if charset_str:
                logger.info(f"Using configured charset for {lang_name}")
            else:
                logger.error(f"No charset found for {lang_name}")
                return None, None, None

        # Load model architecture
        try:
            model = torch.hub.load('baudm/parseq', 'parseq', pretrained=False, trust_repo=True)
            logger.info(f"Model architecture loaded for {lang_name}")
        except Exception as e:
            logger.error(f"Failed to load model architecture: {e}")
            return None, None, None
        
        # Create tokenizer and attach to model
        tokenizer = Tokenizer(charset_str)
        model.tokenizer = tokenizer

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present and handle other key issues
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix
            if k.startswith('module.'):
                k = k[7:]
            # Handle other common prefixes
            if k.startswith('_orig_mod.'):
                k = k[10:]
            new_state_dict[k] = v
        
        # Load state dict with strict=False to handle missing/unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys for {lang_name}: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys for {lang_name}: {unexpected_keys[:5]}...")
        
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
    try:
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
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return "", 0.0

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
    
    /* Make tab text clearly visible */
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
    
    /* Button styling */
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
    
    /* Gallery styling - prevent expansion */
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
    
    /* Box styling */
    .gr-box {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* Primary button styling */
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
""") as demo:
    
    # Header
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
    
    # Footer
    gr.Markdown("""
    ---
    ### 💡 How to use:
    1. **Select a language tab** (Telugu, Bengali, or Oriya)
    2. **Click any sample thumbnail** - it will load into the main preview above
    3. **Click "Extract Text"** button below the preview
    4. **View results** on the right side
    
    ### 📌 Note:
    - Sample thumbnails stay as thumbnails - they don't expand when clicked
    - Only the main preview area changes when you click a sample
    - You can also upload your own images
    """)

# =========================
# Run
# =========================
if __name__ == "__main__":
    # Check directories
    for lang, config in LANGUAGES.items():
        if not os.path.exists(config["model_path"]):
            logger.warning(f"⚠️ Model not found: {config['model_path']} for {lang}")
        if not os.path.exists(config["samples_dir"]):
            os.makedirs(config["samples_dir"], exist_ok=True)
            logger.warning(f"📁 Created samples directory: {config['samples_dir']}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )