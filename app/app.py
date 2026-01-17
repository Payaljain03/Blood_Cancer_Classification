import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Force CPU for Hugging Face deployment
device = torch.device('cpu')

# Load model
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )
    
    # Load weights
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

print("Loading model...")
model = load_model()
print("Model loaded successfully!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_cancer(image):
    """Predicts if blood cell image shows cancer"""
    
    if image is None:
        return {}, "<p>Please upload an image!</p>"
    
    # Preprocess
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs = probabilities.cpu().numpy()[0]
    
    # Apply threshold
    threshold = 0.3
    cancer_prob = probs[1]
    
    if cancer_prob > threshold:
        prediction = "🔴 ALL (Acute Lymphoblastic Leukemia)"
        confidence = cancer_prob * 100
        risk_level = "⚠️ HIGH RISK" if cancer_prob > 0.7 else "⚠️ MODERATE RISK"
        color = "#ff4444"
    else:
        prediction = "🟢 HEM (Normal/Healthy)"
        confidence = probs[0] * 100
        risk_level = "✅ LOW RISK"
        color = "#44ff44"
    
    # Results
    result = {
        "Normal (HEM)": float(probs[0]),
        "Cancer (ALL)": float(probs[1])
    }
    
    interpretation = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f5f5f5; border-left: 5px solid {color};">
        <h2 style="color: {color}; margin-top: 0;">{prediction}</h2>
        <h3 style="color: #333;">Risk Level: {risk_level}</h3>
        <h3 style="color: #333;">Confidence: {confidence:.1f}%</h3>
        
        <hr style="border: 1px solid #ddd; margin: 20px 0;">
        
        <h3>📊 Model Performance Metrics:</h3>
        <ul style="line-height: 1.8;">
            <li><b>Cancer Detection Rate (Recall):</b> 72.2%</li>
            <li><b>Overall Test Accuracy:</b> 66.7%</li>
            <li><b>ROC AUC Score:</b> 0.794</li>
            <li><b>Decision Threshold:</b> 0.3 (optimized for sensitivity)</li>
        </ul>
        
        <h3>⚕️ About This Model:</h3>
        <ul style="line-height: 1.8;">
            <li>Built using <b>ResNet18</b> deep learning architecture</li>
            <li>Trained on blood cell microscopy images</li>
            <li>Tuned to <b>prioritize cancer detection</b> (minimize false negatives)</li>
            <li>Detects <b>94 more cancer cases</b> compared to baseline threshold</li>
        </ul>
        
        <h3>⚠️ Important Disclaimer:</h3>
        <p style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;">
            <b>This is a research demonstration tool only.</b><br>
            NOT intended for clinical diagnosis or medical decision-making.<br>
            Always consult qualified healthcare professionals for medical advice.
        </p>
    </div>
    """
    
    return result, interpretation

# Custom CSS
custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    footer {
        display: none !important;
    }
"""

# Example images
example_images = []
if os.path.exists("example_images"):
    example_images = [
        os.path.join("example_images", f) 
        for f in os.listdir("example_images") 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ]

# Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🩸 Blood Cancer Classification System
    ### AI-Powered Detection of Acute Lymphoblastic Leukemia (ALL)
    
    Upload a blood cell microscopy image to analyze for potential cancer indicators.
    This model uses deep learning to identify abnormal lymphoblasts characteristic of ALL.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="📤 Upload Blood Cell Image", type="numpy")
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📌 Usage Tips:
            - Upload clear microscopy images
            - Supported formats: JPG, PNG, JPEG
            - Processing takes 1-2 seconds
            """)
            
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=input_image,
                    label="💡 Try Example Images"
                )
        
        with gr.Column(scale=1):
            output_label = gr.Label(num_top_classes=2, label="🎯 Prediction Probabilities")
            output_html = gr.HTML(label="📋 Detailed Analysis")
    
    predict_btn.click(
        fn=predict_cancer,
        inputs=input_image,
        outputs=[output_label, output_html]
    )
    
    gr.Markdown("""
    ---
    ### 🔬 About This Project
    
    This classifier was developed using:
    - **Architecture:** ResNet18 with custom classification head
    - **Training Data:** Blood cell microscopy images (ALL vs HEM)
    - **Optimization:** Threshold tuning for improved sensitivity
    - **Framework:** PyTorch + Gradio
    
    ### 📊 Key Features:
    - **High Sensitivity:** 72.2% cancer detection rate
    - **Threshold Optimized:** Reduces missed cancer cases by 15%
    - **Fast Inference:** Results in 1-2 seconds
    
    ### 👨‍💻 Developer
    Built as a research demonstration for blood cancer detection using deep learning.
    
    ⚠️ **Medical Disclaimer:** This tool is for educational and research purposes only. 
    Not validated for clinical use. Always seek professional medical advice.
    """)

# Launch
if __name__ == "__main__":
    demo.launch()