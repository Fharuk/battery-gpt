import streamlit as st
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------------------
# 1. MODEL ARCHITECTURE (Must be identical)
# ----------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class BatteryGPT(nn.Module):
    def __init__(self, feature_size=1, d_model=64, nhead=4, num_layers=2):
        super(BatteryGPT, self).__init__()
        self.input_linear = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src) * math.sqrt(64)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :])

# ----------------------------------------
# 2. APP CONFIGURATION
# ----------------------------------------
st.set_page_config(page_title="BatteryGPT", layout="wide")

st.title("🔋 BatteryGPT: Transformer-Based RUL Predictor")
st.markdown("""
This AI uses a **Nano-Transformer** to predict Lithium-Ion battery degradation.
It was trained on NASA PCoE datasets and demonstrates **Zero-Shot Generalization**.
""")

# ----------------------------------------
# 3. LOAD MODEL
# ----------------------------------------
@st.cache_resource
def load_model():
    model = BatteryGPT()
    # We wrap this in try-except to handle deployment paths
    try:
        # Ensure the .pth file is in the same folder as this app.py
        state_dict = torch.load('battery_gpt_nano.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error("⚠️ Model file 'battery_gpt_nano.pth' not found. Please upload it.")
        return None
    model.eval()
    return model

model = load_model()

# ----------------------------------------
# 4. INTERFACE
# ----------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simulation Settings")
    start_capacity = st.slider("Starting Capacity (Ah)", 1.5, 2.0, 1.85)
    degradation_speed = st.slider("Degradation Noise", 0.001, 0.01, 0.005)
    window_size = 32
    
    run_btn = st.button("🔮 Predict Next Cycle", type="primary")

with col2:
    if run_btn and model:
        # Generate Synthetic Data (Simulating a battery history)
        history = [start_capacity]
        for _ in range(window_size - 1):
            # Create a realistic downward trend with random recovery spikes
            noise = np.random.uniform(0, degradation_speed)
            if np.random.rand() > 0.8: # 20% chance of recovery spike
                next_val = history[-1] + (noise * 0.5)
            else:
                next_val = history[-1] - noise
            history.append(next_val)
            
        # Prepare for AI
        input_seq = torch.Tensor(np.array(history).reshape(1, 32, 1))
        
        # Inference
        with torch.no_grad():
            prediction = model(input_seq).item()
            
        # Visuals
        st.metric("AI Predicted Capacity (Cycle 33)", f"{prediction:.4f} Ah")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, 33), history, label="Input History (32 Cycles)", color='blue', marker='.')
        ax.scatter(33, prediction, color='red', s=100, label="Transformer Prediction", zorder=5)
        
        # Connect the last point
        ax.plot([32, 33], [history[-1], prediction], 'r--')
        
        ax.axhline(y=1.4, color='black', linestyle=':', label="EOL Threshold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Capacity (Ah)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
