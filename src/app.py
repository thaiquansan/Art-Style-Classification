"""
Streamlit App — Art Style Classification
Chạy: streamlit run app.py
Yêu cầu: pip install streamlit torch torchvision Pillow
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ViT_B_16_Weights, ResNet50_Weights
from PIL import Image
import numpy as np
import time
import os

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
NUM_CLASSES = 17
IMG_SIZE    = 224
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Checkpoint paths — sửa lại nếu cần
CKPT = {
    'ArtResNet':           'models/best_ArtResNet.pth',
    'ArtConvGRU':         'models/best_ArtConvGRU.pth',
    'Pretrained ViT':      'models/best_ViT_Pretrained.pth',
    'Pretrained ResNet50': 'models/best_ResNet50_Pretrained.pth',
}

# NOTE:
#   torchvision.datasets.ImageFolder gán index lớp theo thứ tự alphabetical của tên folder.
#   Nếu ART_CLASSES không đúng thứ tự đó, app sẽ dự đoán đúng index nhưng gắn nhãn sai (trật hàng loạt).
#
# Danh sách dưới đây đã được sắp theo đúng alphabetical (khớp ImageFolder cho các folder style này).
ART_CLASSES = [
    'Abstract_Expressionism',
    'Action_painting',
    'Analytical_Cubism',
    'Art_Nouveau_Modern',
    'Color_Field_Painting',
    'Contemporary_Realism',
    'Cubism',
    'Expressionism',
    'Fauvism',
    'Impressionism',
    'Minimalism',
    'Naive_Art_Primitivism',
    'New_Realism',
    'Pop_Art',
    'Post_Impressionism',
    'Symbolism',
    'Synthetic_Cubism',
]

# Màu accent cho từng model
MODEL_COLORS = {
    'ArtResNet':           '#4C9BE8',
    'ArtConvGRU':         '#F5A623',
    'Pretrained ViT':      '#7ED321',
    'Pretrained ResNet50': '#E8534C',
}

# ═══════════════════════════════════════════════════════════
# ============================================================
# 4. MODEL DEFINITIONS
# ============================================================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=nn.GELU):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            act()
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).flatten(1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualSEBlock(nn.Module):
    """
    Residual block + SE.
    Regular enough for small dataset, but still strong for texture/style.
    """
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


def make_stage(in_ch, out_ch, num_blocks, stride=2, dropout=0.0):
    layers = [ResidualSEBlock(in_ch, out_ch, stride=stride, dropout=dropout)]
    for _ in range(num_blocks - 1):
        layers.append(ResidualSEBlock(out_ch, out_ch, stride=1, dropout=dropout))
    return nn.Sequential(*layers)


# ---------- Custom Model 1: ArtResNet ----------
class ArtResNet(nn.Module):
    """
    CNN-centric model for art style:
    - local texture / brush stroke / color patterns
    - multi-scale pooling (avg + max)
    - residual + SE
    """
    def __init__(self, num_classes, dropout_fc=0.30):
        super().__init__()

        # Stem: keep enough low-level detail, but reduce early noise
        self.stem = nn.Sequential(
            ConvBNAct(3, 64, 3, 2),   # 224 -> 112
            ConvBNAct(64, 64, 3, 1),
            ConvBNAct(64, 128, 3, 1),
        )

        # 4 stages, moderate depth
        self.stage1 = make_stage(128, 128, num_blocks=2, stride=1, dropout=0.03)
        self.stage2 = make_stage(128, 256, num_blocks=2, stride=2, dropout=0.06)  # 112 -> 56
        self.stage3 = make_stage(256, 384, num_blocks=2, stride=2, dropout=0.10)  # 56 -> 28
        self.stage4 = make_stage(384, 512, num_blocks=2, stride=2, dropout=0.14)  # 28 -> 14

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(512 * 2),
            nn.Dropout(dropout_fc),
            nn.Linear(512 * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout_fc / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        avg = self.gap(x).flatten(1)
        mx  = self.gmp(x).flatten(1)
        feat = torch.cat([avg, mx], dim=1)
        return self.classifier(feat)


# ---------- Custom Model 2: ArtConvGRU ----------
class ArtConvGRU(nn.Module):
    """
    Hybrid model:
    - CNN stem compresses image to 7x7 feature map
    - 49 tokens only -> GRU stable hơn patch sequence dài
    - BiGRU + attention pooling
    """
    def __init__(self, num_classes, hidden_dim=256, num_layers=2, dropout=0.20):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(3, 64, 3, 2),      # 224 -> 112
            ConvBNAct(64, 128, 3, 2),    # 112 -> 56
            ResidualSEBlock(128, 192, stride=2, dropout=0.03),  # 56 -> 28
            ResidualSEBlock(192, 256, stride=2, dropout=0.05),   # 28 -> 14
            nn.AdaptiveAvgPool2d((7, 7))                         # 14 -> 7x7
        )

        self.embed_dim = 256
        self.num_tokens = 7 * 7  # 49
        self.proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout / 2),
        )
        self.pos_embed = nn.Embedding(self.num_tokens, self.embed_dim)

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        gru_out = hidden_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(gru_out, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out),
            nn.Dropout(dropout),
            nn.Linear(gru_out, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.stem(x)                  # B, 256, 7, 7
        b, c, h, w = feat.shape

        seq = feat.flatten(2).permute(0, 2, 1)   # B, 49, 256
        pos = torch.arange(self.num_tokens, device=x.device)
        seq = self.proj(seq + self.pos_embed(pos).unsqueeze(0))

        out, _ = self.gru(seq)                    # B, 49, 512
        attn_w = F.softmax(self.attn(out), dim=1) # B, 49, 1
        pooled = (out * attn_w).sum(dim=1)        # B, 512

        return self.classifier(pooled)


# ---------- Pretrained wrappers ----------
def get_vit_model(num_classes: int) -> nn.Module:
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    for p in model.parameters():
        p.requires_grad = False

    in_feat = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.LayerNorm(in_feat),
        nn.Dropout(0.20),
        nn.Linear(in_feat, num_classes)
    )

    for p in model.heads.parameters():
        p.requires_grad = True
    return model


def get_resnet50_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights='IMAGENET1K_V2')
    for p in model.parameters():
        p.requires_grad = False

    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.20),
        nn.Linear(in_feat, num_classes)
    )

    for p in model.fc.parameters():
        p.requires_grad = True
    return model

MODEL_BUILDERS = {
    'ArtResNet':           lambda: ArtResNet(NUM_CLASSES),
    'ArtConvGRU':         lambda: ArtConvGRU(NUM_CLASSES),
    'Pretrained ViT':      lambda: get_vit_model(NUM_CLASSES),
    'Pretrained ResNet50': lambda: get_resnet50_model(NUM_CLASSES),
}

# ═══════════════════════════════════════════════════════════
# LOAD MODEL (cache để không load lại mỗi lần)
# ═══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(name):
    path = CKPT[name]
    if not os.path.exists(path):
        return None
    model = MODEL_BUILDERS[name]()
    state = torch.load(path, map_location=DEVICE)
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

# ═══════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(model, img: Image.Image, top_k=5):
    tensor = preprocess(img.convert('RGB')).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    ms = (time.perf_counter() - t0) * 1000
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    top_idx = probs.argsort()[::-1][:top_k]
    return [(ART_CLASSES[i], float(probs[i])) for i in top_idx], ms

def ensemble_logits(models, img):
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    logits_sum = None

    with torch.no_grad():
        for m in models:
            logits = m(tensor)
            logits_sum = logits if logits_sum is None else logits_sum + logits

    probs = F.softmax(logits_sum, dim=1).squeeze().cpu().numpy()
    return probs


# ═══════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ArtVision — Style Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0C0C0F;
    --bg2:       #13131A;
    --bg3:       #1C1C26;
    --border:    #2A2A38;
    --text:      #E8E8F0;
    --muted:     #6B6B80;
    --blue:      #4C9BE8;
    --orange:    #F5A623;
    --green:     #7ED321;
    --red:       #E8534C;
    --gold:      #C9A84C;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Hide default Streamlit header */
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 2rem !important; max-width: 1400px; }

/* Hero title */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #C9A84C 0%, #E8E8F0 50%, #4C9BE8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Model cards */
.model-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.model-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color);
}
.model-name {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.model-pred {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: var(--text);
    margin-bottom: 0.8rem;
}
.model-conf {
    font-size: 0.7rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.bar-track {
    background: var(--bg3);
    border-radius: 4px;
    height: 6px;
    margin-bottom: 0.5rem;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.inference-time {
    font-size: 0.68rem;
    color: var(--muted);
    text-align: right;
    margin-top: 0.5rem;
}
.badge-best {
    display: inline-block;
    background: linear-gradient(135deg, #C9A84C22, #C9A84C44);
    border: 1px solid #C9A84C66;
    color: #C9A84C;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 20px;
    text-transform: uppercase;
    margin-left: 8px;
    vertical-align: middle;
}
.not-loaded {
    color: var(--muted);
    font-size: 0.8rem;
    font-style: italic;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Consensus box */
.consensus-box {
    background: linear-gradient(135deg, #C9A84C11, #4C9BE811);
    border: 1px solid #C9A84C44;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.consensus-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.4rem;
}
.consensus-pred {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: var(--text);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="hero-title">ArtVision</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Art Style Classification · 17 Styles · 4 Models</div>', unsafe_allow_html=True)

# ── Layout: 2 cột ────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

# ── Cột trái: Upload ─────────────────────────────────────────
with col_left:
    st.markdown("#### Upload Artwork")
    uploaded = st.file_uploader(
        "Kéo thả hoặc click để chọn ảnh",
        type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility="collapsed",
    )

    selected_models = st.multiselect(
        "Chọn model để dự đoán",
        options=list(CKPT.keys()),
        default=list(CKPT.keys()),
        help="Có thể chọn nhiều model để so sánh",
    )

    top_k = st.slider("Hiển thị top-K classes", min_value=3, max_value=10, value=5)

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, use_container_width=True, caption=uploaded.name)
        w, h = img.size
        st.markdown(f'<p style="color:var(--muted);font-size:0.72rem;text-align:center">'
                    f'{w}×{h}px · {uploaded.size/1024:.0f} KB</p>', unsafe_allow_html=True)

# ── Cột phải: Kết quả ────────────────────────────────────────
with col_right:
    if not uploaded:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            height:400px; background:var(--bg2); border-radius:16px;
            border:1px dashed var(--border); color:var(--muted); text-align:center;
        ">
            <div style="font-size:3rem; margin-bottom:1rem">🖼️</div>
            <div style="font-size:0.9rem">Upload một bức tranh<br>để bắt đầu phân tích</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        img = Image.open(uploaded).convert('RGB')

        if not selected_models:
            st.warning("Chọn ít nhất 1 model.")
        else:
            # ── Load model 1 lần ──
            models_loaded = {name: load_model(name) for name in selected_models}

            # ── Predict từng model ──
            results = {}
            with st.spinner("Đang phân tích..."):
                for name, model in models_loaded.items():
                    if model is None:
                        results[name] = None
                    else:
                        preds, ms = predict(model, img, top_k=top_k)
                        results[name] = {'preds': preds, 'ms': ms}

            # ── ENSEMBLE (Average Logits) ──
            consensus = None
            ensemble_preds = None

            loaded_models = [m for m in models_loaded.values() if m is not None]

            if loaded_models:
                probs = ensemble_logits(loaded_models, img)

                # top-k ensemble
                top_idx = probs.argsort()[::-1][:top_k]
                ensemble_preds = [(ART_CLASSES[i], probs[i]) for i in top_idx]

                consensus = ART_CLASSES[top_idx[0]]
                conf = probs[top_idx[0]] * 100

                # UI consensus box
                bars_html = ""
                for cls, prob in ensemble_preds:
                    pct = prob * 100
                    bars_html += (
                        f'<div class="model-conf">{cls} <span style="float:right;color:var(--gold)">{pct:.1f}%</span></div>'
                        f'<div class="bar-track">'
                        f'<div class="bar-fill" style="width:{pct:.1f}%;background:#C9A84C99"></div>'
                        f'</div>'
                    )

                st.markdown(f"""
                <div class="consensus-box">
                    <div class="consensus-label">🏛️ Ensemble (Average Logits)</div>
                    <div class="consensus-pred">{consensus}</div>
                    <div style="font-size:0.8rem;color:var(--muted);margin-bottom:0.5rem">
                        {conf:.1f}% confidence
                    </div>
                    {bars_html}
                </div>
                """, unsafe_allow_html=True)

            # ── Grid model cards ──
            pairs = [(selected_models[i], selected_models[i+1] if i+1 < len(selected_models) else None)
                     for i in range(0, len(selected_models), 2)]

            for pair in pairs:
                c1, c2 = st.columns(2, gap="medium")
                for col, model_name in zip([c1, c2], pair):
                    if model_name is None:
                        continue

                    res   = results[model_name]
                    color = MODEL_COLORS.get(model_name, '#AAAAAA')

                    with col:
                        if res is None:
                            st.markdown(f"""
                            <div class="model-card" style="--accent-color:{color}">
                                <div class="model-name">{model_name}</div>
                                <div class="not-loaded">⚠ Checkpoint không tìm thấy<br><code>{CKPT[model_name]}</code></div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            preds     = res['preds']
                            top_class = preds[0][0]
                            top_conf  = preds[0][1]

                            # so với ensemble
                            is_best = (consensus is not None and top_class == consensus)
                            badge   = '<span class="badge-best">✓ ensemble</span>' if is_best else ''

                            bars_html = ""
                            for cls, prob in preds:
                                pct   = prob * 100
                                alpha = '99' if cls == top_class else '55'
                                bars_html += (
                                    f'<div class="model-conf">{cls} <span style="float:right;color:{color}">{pct:.1f}%</span></div>'
                                    f'<div class="bar-track">'
                                    f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color}{alpha}"></div>'
                                    f'</div>'
                                )

                            html = (
                                f'<div class="model-card" style="--accent-color:{color}">'
                                f'<div class="model-name">{model_name}</div>'
                                f'<div class="model-pred">{top_class}{badge}</div>'
                                f'<div style="font-size:0.78rem;color:{color};margin-bottom:0.8rem">{top_conf*100:.1f}% confidence</div>'
                                f'{bars_html}'
                                f'<div class="inference-time">⚡ {res["ms"]:.1f} ms</div>'
                                f'</div>'
                            )
                            st.markdown(html, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="color:var(--muted);font-size:0.7rem;text-align:center;letter-spacing:0.1em">'
    'ARTVISION · WikiArt Dataset · 17 Art Styles · PyTorch</p>',
    unsafe_allow_html=True,
)
