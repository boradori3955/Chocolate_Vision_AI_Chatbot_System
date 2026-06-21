# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, warnings, io, os, zipfile, shutil, tempfile, json, random, re, joblib
from datetime import datetime, timedelta
from pathlib import Path
warnings.filterwarnings("ignore")

# ── 선택적 패키지 임포트 & 누락 감지
def _try_import(pkg_import, pip_name):
    """임포트 시도 후 실패 시 (False, pip_name) 반환"""
    try:
        __import__(pkg_import)
        return True, None
    except ImportError:
        return False, pip_name

# ── 필수 패키지 목록 (import명, pip명)
_REQUIRED_PKGS = [
    ("sklearn",        "scikit-learn"),
    ("imblearn",       "imbalanced-learn"),
    ("xgboost",        "xgboost"),
    ("lightgbm",       "lightgbm"),
    ("catboost",       "catboost"),
    ("joblib",         "joblib"),
]

_missing_pkgs = []
for _imp, _pip in _REQUIRED_PKGS:
    _ok, _pip_name = _try_import(_imp, _pip)
    if not _ok:
        _missing_pkgs.append((_imp, _pip_name))

# 누락 패키지 목록 저장 (탭 진입 시 안내 표시용)
_MISSING_PKG_NAMES = [_p for _, _p in _missing_pkgs]

# 패키지별 실제 임포트
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, roc_curve,
        accuracy_score, precision_score, recall_score, f1_score
    )
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_OK = True
except ImportError:
    _SMOTE_OK = False

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

try:
    from lightgbm import LGBMClassifier
    _LGB_OK = True
except ImportError:
    _LGB_OK = False

try:
    from catboost import CatBoostClassifier
    _CAT_OK = True
except ImportError:
    _CAT_OK = False

st.set_page_config(page_title="🏭 제조 품질 분석 AI 시스템 — 비전", page_icon="🏭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans KR', sans-serif; }
.title-block { background:linear-gradient(135deg,#1a0a00 0%,#2d1200 100%); border-left:4px solid #d4612a; padding:24px 32px; border-radius:4px; margin-bottom:28px; }
.title-block h1 { font-family:'IBM Plex Mono',monospace; color:#f0a060; font-size:1.6rem; margin:0 0 6px 0; }
.title-block p  { color:#888; margin:0; font-size:0.85rem; }
.section-header { font-family:'IBM Plex Mono',monospace; color:#d4612a; font-size:0.73rem; letter-spacing:.15em; text-transform:uppercase; border-bottom:1px solid #2a2a2a; padding-bottom:8px; margin:24px 0 14px 0; }
.stButton > button { background:#d4612a !important; color:#fff !important; border:none !important; border-radius:4px !important; font-family:'IBM Plex Mono',monospace !important; font-size:0.95rem !important; padding:12px 0 !important; width:100% !important; }
.stButton > button:hover { background:#b85020 !important; }
.defect-card { border-radius:6px; padding:14px 18px; margin-bottom:10px; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; }
.metric-card { background:#161616; border:1px solid #2a2a2a; border-radius:6px; padding:16px 12px; text-align:center; }
.metric-card .val { font-family:'IBM Plex Mono',monospace; font-size:1.7rem; font-weight:600; color:#f0a060; }
.metric-card .val.red { color:#e05050; }
.metric-card .lbl { color:#666; font-size:0.75rem; margin-top:4px; }
.proc-badge { display:inline-block; padding:3px 12px; border-radius:3px; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; font-weight:600; margin-bottom:8px; }
.log-box { background:#0a0a0a; border:1px solid #2a2a2a; border-radius:4px; padding:16px; font-family:'IBM Plex Mono',monospace; font-size:0.76rem; color:#aaa; line-height:1.9; max-height:260px; overflow-y:auto; }
/* chat_input 하단 고정 */
[data-testid="stChatInput"] {
    position: fixed; bottom: 1.5rem; left: 50%;
    transform: translateX(-50%);
    width: clamp(320px, 60vw, 860px);
    z-index: 999; border-radius: 0.75rem;
    box-shadow: 0 -2px 16px rgba(0,0,0,0.15); padding: 0.2rem;
}
[data-testid="stChatMessageContainer"], .stChatMessage { padding-bottom: 5.5rem; }
[data-testid="stBottom"] { background: transparent; border-top: none; }
.auto-badge {
    display:inline-block; background:#0d1a0d; border:1px solid #4caf50;
    color:#81c784; border-radius:4px; padding:4px 10px;
    font-family:'IBM Plex Mono',monospace; font-size:0.75rem; margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#0e0e0e","axes.facecolor":"#161616",
    "axes.edgecolor":"#2a2a2a","axes.labelcolor":"#888",
    "xtick.color":"#555","ytick.color":"#555",
    "text.color":"#ccc","grid.color":"#222","grid.linestyle":"--",
})

import matplotlib.font_manager as fm
_korean_fonts = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
for _fp in _korean_fonts:
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
        _fname = fm.FontProperties(fname=_fp).get_name()
        plt.rcParams["font.family"] = _fname
        break
plt.rcParams["axes.unicode_minus"] = False

st.markdown("""
<div class="title-block">
  <h1>🏭 제조 품질 분석 AI 시스템</h1>
  <p>공정 데이터 모델 생성  |  공정 데이터 모델로 불량 예측  |  비전 모델 생성  |  비전 모델로 불량 검사  |  AI 챗봇 질의응답  |  품질 분석 보고서</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📸 AI 학습용 이미지 촬영 가이드", expanded=False):
    st.markdown(
        '<p style="font-size:0.9rem; line-height:2.2; margin:0;">'
        '✅ &nbsp;제품 분류 작업대 위에서 촬영 — 실제 검수 환경과 동일한 배경·조명 유지<br>'
        '📐 &nbsp;분류대 중앙에 제품 놓고 위에서 수직으로, 거리 20~40cm 유지<br>'
        '💡 &nbsp;해상도 640px 이상, 그림자 최소화, 클래스당 30장 이상 수집<br>'
        '⚠️ &nbsp;사무실·창고 등 현장과 다른 곳에서 찍으면 모델 정확도가 크게 낮아집니다.'
        '</p>',
        unsafe_allow_html=True
    )

VER_COLORS    = {"yolov5nu.pt":"#f0e060","yolov8n.pt":"#60c0f0","yolo11n.pt":"#f0a060"}
DEFECT_COLORS = {"Bloom":"#f0e060","Crack":"#f06060","Crack_Bloom":"#f0a060","NoDefects":"#81c784"}

tab_train_image, tab_inspect, tab_chat, tab_report = st.tabs([
    "🤖 비전 모델 생성",
    "🔍 비전 모델로 불량 검사",
    "💬 AI 챗봇 질의응답",
    "📄 비전 품질 분석 보고서",
])


with tab_train_image:
    st.markdown("""<div style="background:#0d1a0d;border-left:4px solid #4caf50;padding:18px 24px;border-radius:4px;margin-bottom:20px">
      <h3 style="color:#81c784;font-family:'IBM Plex Mono',monospace;margin:0 0 6px 0">🤖 비전 모델 생성</h3>
      <p style="color:#666;margin:0;font-size:0.85rem">
        불량 이미지를 업로드하면 비전 AI 모델을 자동으로 생성하고 다운로드할 수 있습니다.
      </p>
    </div>""", unsafe_allow_html=True)

    # 등록된 모델 목록 표시
    if "selected_model" in st.session_state:
        sm = st.session_state["selected_model"]
        mc = VER_COLORS.get(sm["model"], "#4caf50")
        badge = "📂 업로드됨" if sm.get("uploaded") else "🤖 학습됨"
        st.markdown(f"""
        <div style="background:#161616;border:1px solid {mc};border-radius:6px;
                    padding:12px 16px;margin-bottom:8px">
          <span style="color:{mc};font-family:monospace;font-weight:600">{badge} {sm['model']}</span>
          &nbsp;&nbsp;
          <span style="color:#888;font-size:0.85rem">mAP50: <b style="color:#ccc">{sm['mAP50']*100:.1f}%</b></span>
          &nbsp;&nbsp;
          <span style="color:#888;font-size:0.85rem">속도: <b style="color:#ccc">{sm['speed_ms']:.1f}ms</b></span>
        </div>
        """, unsafe_allow_html=True)


    # ── 모델 생성 방식 선택 (Classification / Detection)
    st.markdown("""
    <style>
    div[data-testid="stRadio"] > div { gap: 60px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🎯 학습 방식 선택</div>', unsafe_allow_html=True)
    train_mode = st.radio(
        "학습 방식을 선택하세요",
        ["Classification (EfficientNet, MobileNetV3, ResNet)", "Detection (YOLO)"],
        horizontal=True,
        key="train_mode"
    )

    if train_mode.startswith("Classification"):


        # ── Step 1. 학습 이미지 업로드
        st.markdown("### Step 1. 이미지 업로드")
        col_cls1, col_cls2 = st.columns(2)
        with col_cls1:
            normal_imgs = st.file_uploader(
                "✅ 정상 이미지 업로드 (여러 장 선택 가능)",
                type=["jpg","jpeg","png"],
                accept_multiple_files=True,
                key="cls_normal_images"
            )
            if normal_imgs:
                st.success(f"정상 이미지 {len(normal_imgs)}장 업로드됨")
        with col_cls2:
            defect_imgs = st.file_uploader(
                "❌ 불량 이미지 업로드 (여러 장 선택 가능)",
                type=["jpg","jpeg","png"],
                accept_multiple_files=True,
                key="cls_defect_images"
            )
            if defect_imgs:
                st.error(f"불량 이미지 {len(defect_imgs)}장 업로드됨")

        if normal_imgs and defect_imgs:
            total_imgs = len(normal_imgs) + len(defect_imgs)
            n_val_n = max(1, int(len(normal_imgs) * 0.2))
            n_val_d = max(1, int(len(defect_imgs) * 0.2))
            st.markdown(
                f'<div style="background:#161616;border:1px solid #2a2a2a;border-radius:4px;'
                f'padding:12px;font-size:0.82rem;color:#888;margin-bottom:8px">'
                f'📊 총 <b style="color:#ccc">{total_imgs}장</b> 로드 완료  |  '
                f'train: <b style="color:#81c784">{len(normal_imgs)-n_val_n}정상 + {len(defect_imgs)-n_val_d}불량</b>  |  '
                f'val: <b style="color:#f0a060">{n_val_n}정상 + {n_val_d}불량</b>'
                f'</div>', unsafe_allow_html=True
            )

            # ── Step 2. 학습 파라미터 설정
            st.markdown("### Step 2. 학습 설정")
            col_cs1, col_cs2, col_cs3 = st.columns(3)
            with col_cs1:
                cls_epochs = st.slider("Epochs", 5, 50, 10, 5, key="cls_epochs")
            with col_cs2:
                cls_lr = st.selectbox("Learning Rate", [0.001, 0.0005, 0.0001], index=0, key="cls_lr")
            with col_cs3:
                cls_batch = st.selectbox("배치 크기", [8, 16, 32], index=1, key="cls_batch",
                                         help="GPU 메모리 부족하면 8로 설정")

            CLS_MODEL_OPTIONS = ["EfficientNet-B0", "MobileNetV3-Small", "ResNet-18"]
            CLS_COLORS = {
                "EfficientNet-B0":   "#f0e060",
                "MobileNetV3-Small": "#60c0f0",
                "ResNet-18":         "#f0a060",
            }
            CLS_DESC = {
                "EfficientNet-B0":   "정확도/속도 균형 최고 | 20MB | 소량 데이터에 강함",
                "MobileNetV3-Small": "가장 빠름 | 10MB | 엣지 배포 최적",
                "ResNet-18":         "안정적 파인튜닝 | 45MB | 검증된 성능",
            }
            cls_models_to_run = st.multiselect(
                "비교할 모델",
                CLS_MODEL_OPTIONS,
                default=CLS_MODEL_OPTIONS,
                key="cls_models",
                format_func=lambda x: f"{x}  —  {CLS_DESC[x]}"
            )

            if cls_models_to_run:
                est_cls = {"EfficientNet-B0": 3, "MobileNetV3-Small": 2, "ResNet-18": 3}
                st.info(f"⏱️ 예상 시간: 약 **{sum(est_cls.get(m,3) for m in cls_models_to_run) * cls_epochs // 10}분** "
                        f"(이미지 {total_imgs}장 / {cls_epochs} epochs 기준)")

            # ── Step 3. 모델 학습 실행
            st.markdown("### Step 3. 학습 실행")

            if st.button("🚀 Classification 학습 시작", key="cls_train_btn") and cls_models_to_run:
                # 필수 패키지 설치 여부 확인
                try:
                    import torch, torchvision
                    from torchvision import transforms, models
                    from torch.utils.data import DataLoader, Dataset
                    from PIL import Image as _PIL_Image
                except ImportError as _ie:
                    st.error(f"❌ `pip install torch torchvision pillow` 를 먼저 실행하세요. ({_ie})")
                    st.stop()

                import random as _rnd
                _rnd.seed(42)
                torch.manual_seed(42)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # ── 데이터셋 클래스
                class _BinaryDataset(Dataset):
                    def __init__(self, items, transform):
                        self.items = items      # [(pil_img, label), ...]
                        self.transform = transform
                    def __len__(self): return len(self.items)
                    def __getitem__(self, idx):
                        img, label = self.items[idx]
                        return self.transform(img), label

                # ── 이미지 로드 → PIL
                def _load_pil(uploaded_files):
                    imgs = []
                    for f in uploaded_files:
                        try:
                            img = _PIL_Image.open(f).convert("RGB")
                            imgs.append(img)
                        except Exception:
                            pass
                        f.seek(0)
                    return imgs

                cls_status_ph = st.empty()
                cls_prog_bar  = st.progress(0)

                cls_status_ph.markdown(
                    '<div style="background:#161616;border:1px solid #f0e060;border-radius:4px;'
                    'padding:12px;color:#f0e060;font-family:monospace">📂 이미지 로딩 중...</div>',
                    unsafe_allow_html=True
                )

                normal_pils = _load_pil(normal_imgs)
                defect_pils = _load_pil(defect_imgs)

                _rnd.shuffle(normal_pils); _rnd.shuffle(defect_pils)
                n_val_n2 = max(1, int(len(normal_pils) * 0.2))
                n_val_d2 = max(1, int(len(defect_pils) * 0.2))

                # 레이블: 0=정상, 1=불량
                train_items = (
                    [(img, 0) for img in normal_pils[n_val_n2:]] +
                    [(img, 1) for img in defect_pils[n_val_d2:]]
                )
                val_items = (
                    [(img, 0) for img in normal_pils[:n_val_n2]] +
                    [(img, 1) for img in defect_pils[:n_val_d2]]
                )

                train_tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                ])
                val_tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
                ])

                train_loader = DataLoader(_BinaryDataset(train_items, train_tf),
                                          batch_size=cls_batch, shuffle=True)
                val_loader   = DataLoader(_BinaryDataset(val_items, val_tf),
                                          batch_size=cls_batch, shuffle=False)

                # ── 모델 빌더
                def _build_model(name):
                    if name == "EfficientNet-B0":
                        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
                    elif name == "MobileNetV3-Small":
                        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 2)
                    else:  # ResNet-18
                        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                        m.fc = torch.nn.Linear(m.fc.in_features, 2)
                    return m.to(device)

                cls_results   = []
                n_cls_models  = len(cls_models_to_run)
                cls_tmp       = tempfile.mkdtemp()

                for i, model_name in enumerate(cls_models_to_run):
                    mcolor = CLS_COLORS.get(model_name, "#f0a060")
                    cls_status_ph.markdown(
                        f'<div style="background:#161616;border:1px solid {mcolor};border-radius:4px;'
                        f'padding:12px;color:{mcolor};font-family:monospace">'
                        f'⚙️ [{i+1}/{n_cls_models}] <b>{model_name}</b> 파인튜닝 중 (device: {device})...</div>',
                        unsafe_allow_html=True
                    )
                    try:
                        t0    = time.time()
                        mdl   = _build_model(model_name)
                        opt   = torch.optim.Adam(mdl.parameters(), lr=cls_lr)
                        crit  = torch.nn.CrossEntropyLoss()
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cls_epochs)

                        best_val_acc = 0.0
                        best_state   = None
                        epoch_log    = []

                        ep_ph = st.empty()
                        for ep in range(cls_epochs):
                            # train
                            mdl.train()
                            for xb, yb in train_loader:
                                xb, yb = xb.to(device), yb.to(device)
                                opt.zero_grad()
                                loss = crit(mdl(xb), yb)
                                loss.backward()
                                opt.step()
                            sched.step()

                            # val
                            mdl.eval()
                            correct = total = 0
                            with torch.no_grad():
                                for xb, yb in val_loader:
                                    xb, yb = xb.to(device), yb.to(device)
                                    preds = mdl(xb).argmax(1)
                                    correct += (preds == yb).sum().item()
                                    total   += yb.size(0)
                            val_acc = correct / total if total > 0 else 0.0
                            epoch_log.append(val_acc)

                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_state   = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}

                            ep_ph.markdown(
                                f'<div style="font-size:0.8rem;color:#888;font-family:monospace">'
                                f'  Epoch {ep+1}/{cls_epochs}  |  val_acc: <b style="color:#81c784">{val_acc*100:.1f}%</b>'
                                f'  |  best: <b style="color:#f0e060">{best_val_acc*100:.1f}%</b></div>',
                                unsafe_allow_html=True
                            )

                        ep_ph.empty()
                        elapsed = time.time() - t0

                        # best 모델 저장 (.pth)
                        mdl.load_state_dict(best_state)
                        save_path = os.path.join(cls_tmp, f"best_{model_name.replace('-','_').replace(' ','_')}.pth")
                        torch.save({"model_name": model_name, "state_dict": best_state, "classes": ["normal","defect"]}, save_path)

                        # 추론 속도 측정 (CPU 기준 10회 평균)
                        mdl_cpu = mdl.cpu().eval()
                        dummy   = torch.randn(1, 3, 224, 224)
                        t_sp    = time.time()
                        with torch.no_grad():
                            for _ in range(10): mdl_cpu(dummy)
                        speed_ms = (time.time() - t_sp) / 10 * 1000
                        size_mb  = os.path.getsize(save_path) / (1024*1024)

                        cls_results.append({
                            "model":      model_name,
                            "acc":        best_val_acc,
                            "speed_ms":   speed_ms,
                            "size_mb":    size_mb,
                            "train_min":  elapsed / 60,
                            "save_path":  save_path,
                            "epoch_log":  epoch_log,
                            "success":    True,
                        })

                    except Exception as e:
                        import traceback
                        cls_results.append({"model": model_name, "success": False, "error": str(e)})
                        st.warning(f"⚠️ {model_name} 실패: {e}")

                    cls_prog_bar.progress(int((i+1)/n_cls_models*100))

                cls_status_ph.markdown(
                    '<div style="background:#0d1a0d;border:1px solid #4caf50;border-radius:4px;'
                    'padding:12px;color:#81c784;font-family:monospace">🎉 모든 Classification 모델 학습 완료!</div>',
                    unsafe_allow_html=True
                )
                st.session_state["cls_results"] = cls_results
                st.session_state["cls_tmp"]     = cls_tmp

            # ── Step 4. 모델 성능 비교
            if "cls_results" in st.session_state:
                cls_ok = [r for r in st.session_state["cls_results"] if r.get("success")]
                if not cls_ok:
                    st.error("성공한 학습 결과가 없습니다.")
                else:
                    CLS_COLORS2 = {
                        "EfficientNet-B0":   "#f0e060",
                        "MobileNetV3-Small": "#60c0f0",
                        "ResNet-18":         "#f0a060",
                    }
                    st.markdown("### Step 4. 모델 비교 결과")
                    df_cls = pd.DataFrame([{
                        "모델":       r["model"],
                        "정확도(Val)": f"{r['acc']*100:.1f}%",
                        "속도(ms/img)": f"{r['speed_ms']:.1f}",
                        "크기(MB)":   f"{r['size_mb']:.1f}",
                        "학습시간":   f"{r['train_min']:.1f}분",
                    } for r in cls_ok])
                    st.dataframe(df_cls, use_container_width=True, hide_index=True)

                    # 학습 곡선 및 성능 막대 차트 렌더링
                    _names_c  = [r["model"] for r in cls_ok]
                    _colors_c = [CLS_COLORS2.get(r["model"], "#f0a060") for r in cls_ok]

                    n_plots = 2 + len(cls_ok)
                    fig_cls, axes_cls = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
                    fig_cls.tight_layout(pad=3.0)

                    # 정확도 비교 막대 차트
                    bars = axes_cls[0].bar(_names_c, [r["acc"]*100 for r in cls_ok],
                                           color=_colors_c, edgecolor="#2a2a2a", width=0.5)
                    axes_cls[0].set_title("Val 정확도 (%)", fontsize=9, pad=6)
                    axes_cls[0].set_ylim(0, 110)
                    axes_cls[0].set_xticks(range(len(_names_c)))
                    axes_cls[0].set_xticklabels(_names_c, rotation=20, ha="right", fontsize=7)
                    axes_cls[0].grid(True, alpha=0.3, axis="y")
                    for bar, r in zip(bars, cls_ok):
                        axes_cls[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                                         f"{r['acc']*100:.1f}%", ha="center", va="bottom", fontsize=8, color="#ccc")

                    # 추론 속도 비교 막대 차트
                    bars2 = axes_cls[1].bar(_names_c, [r["speed_ms"] for r in cls_ok],
                                            color=_colors_c, edgecolor="#2a2a2a", width=0.5)
                    axes_cls[1].set_title("추론 속도 (ms)", fontsize=9, pad=6)
                    axes_cls[1].set_xticks(range(len(_names_c)))
                    axes_cls[1].set_xticklabels(_names_c, rotation=20, ha="right", fontsize=7)
                    axes_cls[1].grid(True, alpha=0.3, axis="y")
                    for bar, r in zip(bars2, cls_ok):
                        axes_cls[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                                         f"{r['speed_ms']:.1f}", ha="center", va="bottom", fontsize=8, color="#ccc")

                    # 모델별 학습 곡선
                    for ax, r in zip(axes_cls[2:], cls_ok):
                        mc = CLS_COLORS2.get(r["model"], "#f0a060")
                        ax.plot(range(1, len(r["epoch_log"])+1), [v*100 for v in r["epoch_log"]],
                                color=mc, linewidth=2, marker="o", markersize=3)
                        ax.set_title(f"{r['model']} 학습곡선", fontsize=8, pad=6)
                        ax.set_xlabel("Epoch", fontsize=7)
                        ax.set_ylabel("Val Acc (%)", fontsize=7)
                        ax.set_ylim(0, 110)
                        ax.grid(True, alpha=0.3)

                    st.pyplot(fig_cls); plt.close()

                    # 최적 모델 추천 카드
                    best_acc_c   = max(cls_ok, key=lambda x: x["acc"])
                    best_speed_c = min(cls_ok, key=lambda x: x["speed_ms"])
                    best_bal_c   = max(cls_ok, key=lambda x: x["acc"] - x["speed_ms"]*0.001)
                    col_cr1, col_cr2, col_cr3 = st.columns(3)
                    for col, r, icon, label, sub, brd in [
                        (col_cr1, best_acc_c,   "🏆", "정확도 최고", f"Val Acc: {best_acc_c['acc']*100:.1f}%",    "#4caf50"),
                        (col_cr2, best_speed_c, "⚡", "속도 최고",   f"속도: {best_speed_c['speed_ms']:.1f}ms",   "#2196f3"),
                        (col_cr3, best_bal_c,   "⚖️", "균형 최고",   f"Val Acc: {best_bal_c['acc']*100:.1f}%",    "#f0a060"),
                    ]:
                        with col:
                            vc = CLS_COLORS2.get(r["model"], brd)
                            st.markdown(f"""
                            <div style="background:#111;border:1px solid {vc};border-radius:6px;
                                        padding:16px;text-align:center;margin-bottom:8px">
                              <div style="color:{vc};font-size:0.75rem;margin-bottom:6px">{icon} {label}</div>
                              <div style="color:#fff;font-family:monospace;font-size:1.1rem;font-weight:600">{r['model']}</div>
                              <div style="color:{vc};margin-top:4px">{sub}</div>
                              <div style="color:#555;font-size:0.7rem;margin-top:6px">크기: {r['size_mb']:.1f}MB  |  학습: {r['train_min']:.1f}분</div>
                            </div>""", unsafe_allow_html=True)

                    # ── Step 5. 모델 선택 및 다운로드
                    st.markdown("### Step 5. 모델 선택 & 다운로드")
                    cls_model_map     = {r["model"]: r for r in cls_ok}
                    cls_selected_name = st.selectbox(
                        "적용할 모델 선택", list(cls_model_map.keys()),
                        format_func=lambda x: (
                            f"{x}  |  Val Acc: {cls_model_map[x]['acc']*100:.1f}%"
                            f"  |  속도: {cls_model_map[x]['speed_ms']:.1f}ms"
                            f"  |  크기: {cls_model_map[x]['size_mb']:.1f}MB"
                        ),
                        key="cls_select"
                    )
                    cls_sel = cls_model_map[cls_selected_name]

                    st.session_state["selected_model"] = {
                        "model":    cls_sel["model"],
                        "short":    cls_sel["model"].replace("-","_").replace(" ","_"),
                        "mAP50":    cls_sel["acc"],
                        "speed_ms": cls_sel["speed_ms"],
                        "size_mb":  cls_sel["size_mb"],
                        "best_pt":  cls_sel["save_path"],
                        "mode":     "classification",
                    }

                    if os.path.exists(cls_sel["save_path"]):
                        with open(cls_sel["save_path"], "rb") as f:
                            st.download_button(
                                label=f"⬇ {cls_selected_name} (.pth) 다운로드",
                                data=f.read(),
                                file_name=f"best_{cls_sel['model'].replace(' ','_')}.pth",
                                mime="application/octet-stream",
                                use_container_width=True,
                                key="cls_dl_pth"
                            )
                    st.info("💡 .pth 파일은 torch.load()로 바로 불러올 수 있습니다. "
                            "ONNX 변환이 필요하면 별도 스크립트를 사용하세요.")

        elif normal_imgs or defect_imgs:
            st.warning("⚠️ 정상 이미지와 불량 이미지를 **모두** 업로드해야 학습할 수 있습니다.")
        else:
            st.info("👆 정상/불량 이미지 파일을 업로드하세요.")

    else:

        st.markdown("### Step 1. ZIP 업로드")
        uploaded_zip = st.file_uploader("데이터셋 ZIP 업로드", type=["zip"], key="yolo_zip")

        if not uploaded_zip:
            st.info("👆 ZIP 파일을 업로드하세요.")
        else:
            st.markdown("### Step 2. 학습 설정")    
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.slider("Epochs", 10, 100, 50, 10)
            with col2:
                imgsz  = st.selectbox("이미지 크기", [416, 640, 832], index=1)
            with col3:
                batch  = st.selectbox("배치 크기", [4, 8, 16], index=0,
                                      help="GPU 메모리 부족하면 4로 설정")

            models_to_run = st.multiselect(
                "비교할 모델",
                ["yolov5nu.pt", "yolov8n.pt", "yolo11n.pt"],
                default=["yolov5nu.pt", "yolov8n.pt", "yolo11n.pt"],
            )

            if models_to_run:
                est = {"yolov5nu.pt": 15, "yolov8n.pt": 18, "yolo11n.pt": 20}
                st.info(f"⏱️ 예상 시간: 약 **{sum(est.get(m, 20) for m in models_to_run)}분**")

            st.markdown("### Step 3. 학습 실행")

            if st.button("🚀 학습 시작", key="yolo_train_btn") and models_to_run:
                try:
                    from ultralytics import YOLO
                except ImportError:
                    st.error("❌ `pip install ultralytics` 를 먼저 실행하세요.")
                    st.stop()

                tmp_dir  = tempfile.mkdtemp()
                zip_path = os.path.join(tmp_dir, "dataset.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.read())
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_dir)

                yaml_files = list(Path(tmp_dir).rglob("data.yaml"))
                if not yaml_files:
                    st.error("❌ data.yaml 파일을 찾을 수 없습니다.")
                    shutil.rmtree(tmp_dir)
                    st.stop()

                data_yaml  = str(yaml_files[0])
                output_dir = os.path.join(tmp_dir, "compare")
                os.makedirs(output_dir, exist_ok=True)
                st.success("✅ 데이터셋 확인 완료")

                status_ph  = st.empty()
                prog_bar   = st.progress(0)
                yolo_results = []
                n_models   = len(models_to_run)

                for i, model_name in enumerate(models_to_run):
                    short  = model_name.replace(".pt", "")
                    mcolor = VER_COLORS.get(model_name, "#4caf50")
                    status_ph.markdown(
                        f'<div style="background:#161616;border:1px solid {mcolor};border-radius:4px;'
                        f'padding:12px;color:{mcolor};font-family:monospace">'
                        f'⚙️ [{i+1}/{n_models}] <b>{model_name}</b> 학습 중... (완료까지 기다려주세요)'
                        f'</div>', unsafe_allow_html=True)
                    try:
                        t0     = time.time()
                        model  = YOLO(model_name)
                        result = model.train(
                            data=data_yaml, epochs=epochs, imgsz=imgsz,
                            batch=batch, project=output_dir, name=short,
                            exist_ok=True, verbose=False,
                        )
                        elapsed  = time.time() - t0
                        metrics  = result.results_dict
                        best_pt  = os.path.join(output_dir, short, "weights", "best.pt")
                        size_mb  = os.path.getsize(best_pt) / (1024*1024) if os.path.exists(best_pt) else 0

                        yolo_results.append({
                            "model":     model_name,
                            "short":     short,
                            "mAP50":     metrics.get("metrics/mAP50(B)",     0.0),
                            "mAP50-95":  metrics.get("metrics/mAP50-95(B)",  0.0),
                            "precision": metrics.get("metrics/precision(B)",  0.0),
                            "recall":    metrics.get("metrics/recall(B)",     0.0),
                            "speed_ms":  result.speed.get("inference", 0.0),
                            "size_mb":   size_mb,
                            "train_min": elapsed / 60,
                            "best_pt":   best_pt,
                            "success":   True,
                        })
                    except Exception as e:
                        yolo_results.append({"model": model_name, "success": False, "error": str(e)})
                        st.warning(f"⚠️ {model_name} 실패: {e}")

                    prog_bar.progress(int((i+1) / n_models * 100))

                status_ph.markdown(
                    '<div style="background:#0d1a0d;border:1px solid #4caf50;border-radius:4px;'
                    'padding:12px;color:#81c784;font-family:monospace">🎉 모든 모델 학습 완료!</div>',
                    unsafe_allow_html=True)

                st.session_state["yolo_results"] = yolo_results
                st.session_state["yolo_imgsz"]   = imgsz
                st.session_state["yolo_tmp_dir"] = tmp_dir

            if "yolo_results" in st.session_state:
                success_r = [r for r in st.session_state["yolo_results"] if r.get("success")]
                if not success_r:
                    st.error("성공한 학습 결과가 없습니다.")
                else:
                    st.markdown("### Step 4. 모델 비교 결과")
                    df_res = pd.DataFrame([{
                        "모델":      r["model"],
                        "mAP50":    f"{r['mAP50']*100:.1f}%",
                        "mAP50-95": f"{r['mAP50-95']*100:.1f}%",
                        "정밀도":   f"{r['precision']*100:.1f}%",
                        "재현율":   f"{r['recall']*100:.1f}%",
                        "속도(ms)": f"{r['speed_ms']:.1f}",
                        "크기(MB)": f"{r['size_mb']:.1f}",
                        "학습시간": f"{r['train_min']:.1f}분",
                    } for r in success_r])
                    st.dataframe(df_res, use_container_width=True, hide_index=True)

                    names_y  = [r["model"] for r in success_r]
                    colors_y = [VER_COLORS.get(r["model"], "#f0a060") for r in success_r]
                    metric_data = {
                        "mAP50 (%)":    [r["mAP50"]*100    for r in success_r],
                        "mAP50-95 (%)": [r["mAP50-95"]*100 for r in success_r],
                        "정밀도 (%)":   [r["precision"]*100 for r in success_r],
                        "재현율 (%)":   [r["recall"]*100    for r in success_r],
                        "속도 (ms)":    [r["speed_ms"]      for r in success_r],
                    }
                    fig_bar, axes_bar = plt.subplots(1, 5, figsize=(18, 4))
                    fig_bar.tight_layout(pad=3.0)
                    for ax, (title, vals) in zip(axes_bar, metric_data.items()):
                        bars = ax.bar(names_y, vals, color=colors_y, edgecolor='#2a2a2a', width=0.5)
                        ax.set_title(title, fontsize=9, pad=6)
                        ax.set_ylim(0, max(vals)*1.2 if max(vals) > 0 else 1)
                        ax.set_xticks(range(len(names_y)))
                        ax.set_xticklabels(names_y, rotation=25, ha='right', fontsize=7)
                        ax.grid(True, alpha=0.3, axis='y')
                        for bar, v in zip(bars, vals):
                            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                                    f'{v:.1f}', ha='center', va='bottom', fontsize=7, color='#ccc')
                    st.pyplot(fig_bar); plt.close()

                    radar_keys   = ["mAP50", "mAP50-95", "precision", "recall"]
                    radar_labels = ["mAP50", "mAP50-95", "정밀도", "재현율"]
                    angles = np.linspace(0, 2*np.pi, len(radar_keys), endpoint=False).tolist()
                    angles += angles[:1]
                    fig_r, ax_r = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                    ax_r.set_facecolor('#161616')
                    ax_r.set_theta_offset(np.pi/2); ax_r.set_theta_direction(-1)
                    ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(radar_labels, fontsize=10)
                    ax_r.set_ylim(0, 1.0); ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
                    ax_r.set_yticklabels(['0.25','0.5','0.75','1.0'], fontsize=7, color='#555')
                    ax_r.grid(color='#2a2a2a', linewidth=0.8)
                    for r in success_r:
                        vals_r  = [r[k] for k in radar_keys] + [r[radar_keys[0]]]
                        color_r = VER_COLORS.get(r["model"], "#f0a060")
                        ax_r.plot(angles, vals_r, color=color_r, linewidth=2, label=r["model"])
                        ax_r.fill(angles, vals_r, color=color_r, alpha=0.08)
                    ax_r.legend(fontsize=8, facecolor='#161616', edgecolor='#333',
                                labelcolor='#ccc', loc='upper right', bbox_to_anchor=(1.35, 1.1))
                    fig_r.patch.set_facecolor('#0e0e0e'); fig_r.tight_layout()
                    st.pyplot(fig_r); plt.close()

                    best_acc   = max(success_r, key=lambda x: x["mAP50"])
                    best_speed = min(success_r, key=lambda x: x["speed_ms"])
                    best_bal   = max(success_r, key=lambda x: (x["mAP50"] + (1/max(x["speed_ms"],0.1))*10))
                    col_r1, col_r2, col_r3 = st.columns(3)
                    for col, r, icon, label, sub, border in [
                        (col_r1, best_acc,   "🏆", "정확도 최고", f"mAP50: {best_acc['mAP50']*100:.1f}%",  "#4caf50"),
                        (col_r2, best_speed, "⚡", "속도 최고",   f"속도: {best_speed['speed_ms']:.1f}ms",  "#2196f3"),
                        (col_r3, best_bal,   "⚖️", "균형 최고",   f"mAP50: {best_bal['mAP50']*100:.1f}%",   "#f0a060"),
                    ]:
                        with col:
                            vc = VER_COLORS.get(r["model"], border)
                            st.markdown(f"""
                            <div style="background:#111;border:1px solid {vc};border-radius:6px;
                                        padding:16px;text-align:center;margin-bottom:8px">
                              <div style="color:{vc};font-size:0.75rem;margin-bottom:6px">{icon} {label}</div>
                              <div style="color:#fff;font-family:monospace;font-size:1.1rem;font-weight:600">{r['model']}</div>
                              <div style="color:{vc};margin-top:4px">{sub}</div>
                              <div style="color:#555;font-size:0.7rem;margin-top:6px">크기: {r['size_mb']:.1f}MB  |  학습: {r['train_min']:.1f}분</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("### Step 5. 모델 선택 & 다운로드")
                    model_map     = {r["model"]: r for r in success_r}
                    selected_name = st.selectbox(
                        "적용할 모델 선택", list(model_map.keys()),
                        format_func=lambda x: (
                            f"{x}  |  mAP50: {model_map[x]['mAP50']*100:.1f}%"
                            f"  |  속도: {model_map[x]['speed_ms']:.1f}ms"
                            f"  |  크기: {model_map[x]['size_mb']:.1f}MB"
                        )
                    )
                    selected = model_map[selected_name]
                    _imgsz   = st.session_state.get("yolo_imgsz", 640)
                    st.session_state["selected_model"] = selected

                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        if os.path.exists(selected["best_pt"]):
                            with open(selected["best_pt"], "rb") as f:
                                st.download_button(
                                    label=f"⬇ {selected_name} (.pt) 다운로드",
                                    data=f.read(),
                                    file_name=f"best_{selected['short']}.pt",
                                    mime="application/octet-stream",
                                    use_container_width=True,
                                )
                    with col_d2:
                        if st.button("📦 ONNX 변환 후 다운로드", use_container_width=True, key="onnx_btn"):
                            with st.spinner("ONNX 변환 중..."):
                                try:
                                    from ultralytics import YOLO as _YOLO
                                    m = _YOLO(selected["best_pt"])
                                    m.export(format="onnx", imgsz=_imgsz, simplify=True)
                                    onnx_path = selected["best_pt"].replace(".pt", ".onnx")
                                    if os.path.exists(onnx_path):
                                        with open(onnx_path, "rb") as f:
                                            st.download_button(
                                                label=f"⬇ {selected_name} (.onnx) 다운로드",
                                                data=f.read(),
                                                file_name=f"best_{selected['short']}.onnx",
                                                mime="application/octet-stream",
                                                use_container_width=True,
                                                key="onnx_dl"
                                            )
                                        st.success("✅ weights/best.onnx 로 교체하면 바로 적용됩니다!")
                                except Exception as e:
                                    st.error(f"ONNX 변환 실패: {e}")


# ────────────────────────────────────────────────────────────
# TAB: 비전 모델로 불량 검사
# ────────────────────────────────────────────────────────────
with tab_inspect:
    # ── 비전 모델 생성 탭에서 전달된 모델 모드 판별
    _t2_inh    = st.session_state.get("selected_model")
    _t2_mode   = _t2_inh.get("mode", "detection") if _t2_inh else "detection"
    _t2_is_cls = (_t2_mode == "classification")

    if _t2_is_cls:
        _desc_txt = "이미지를 업로드하면 정상/불량 여부를 자동으로 판정하고 결과를 저장합니다."
    else:
        _desc_txt = "생성된 비전 모델로 불량 유형을 자동으로 감지하고 결과를 저장합니다."

    st.markdown(
        '<div style="background:#1a0d1a;border-left:4px solid #c060f0;padding:18px 24px;'
        'border-radius:4px;margin-bottom:20px">'
        '<h3 style="color:#d090f0;font-family:monospace;margin:0 0 6px 0">🔍 비전 모델로 불량 검사</h3>'
        f'<p style="color:#666;margin:0;font-size:0.85rem">{_desc_txt}</p>'
        '</div>',
        unsafe_allow_html=True
    )

    col_u1, col_u2, col_u3 = st.columns(3)

    # ── ① 검사 모델 업로드
    with col_u1:
        if _t2_is_cls:
            st.markdown("**① Classification 모델 (.pth)**")
        else:
            st.markdown("**① YOLO 모델 (.pt / .onnx)**")

        inherited_model = st.session_state.get("selected_model")

        if inherited_model and inherited_model.get("best_pt") and os.path.exists(inherited_model["best_pt"]):
            _acc_val = inherited_model.get("mAP50", 0) * 100
            _acc_lbl = "Val Acc" if _t2_is_cls else "mAP50"
            _badge   = "📂 업로드됨" if inherited_model.get("uploaded") else "🤖 비전 모델 생성 탭에서 학습됨"
            st.markdown(
                f'<div class="auto-badge">✅ {_badge}: {inherited_model["model"]} '
                f'({_acc_lbl} {_acc_val:.1f}%)</div>',
                unsafe_allow_html=True
            )
            model_file = None
            _use_inherited_model = True
        else:
            if _t2_is_cls:
                model_file = st.file_uploader("모델 파일 (.pth)", type=["pth"], key="t2_model")
            else:
                model_file = st.file_uploader("모델 파일 (.pt / .onnx)", type=["pt","onnx"], key="t2_model")
            _use_inherited_model = False

    # ── ② 검사 이미지 업로드
    with col_u2:
        st.markdown("**② 이미지 업로드** (다중 선택)")

        # 실행 경로의 images/ 폴더 자동 탐색
        if "t2_auto_images" not in st.session_state and not st.session_state.get("t2_img_manual_mode"):
            _img_dir = Path(__file__).parent / "images"
            _exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            _auto_paths = sorted([p for p in _img_dir.iterdir() if p.suffix.lower() in _exts]) \
                if _img_dir.is_dir() else []
            st.session_state["t2_auto_images"] = _auto_paths

        _auto_paths = st.session_state.get("t2_auto_images", [])

        if _auto_paths and not st.session_state.get("t2_img_manual_mode"):
            st.markdown(
                f'<div class="auto-badge">✅ images/ 폴더 자동 로드 ({len(_auto_paths)}개)</div>',
                unsafe_allow_html=True
            )
            if st.button("🔄 다른 이미지 업로드", key="t2_img_reset", use_container_width=True):
                st.session_state.pop("t2_auto_images", None)
                st.session_state["t2_img_manual_mode"] = True
                st.rerun()
            # Path 객체를 file-like 래퍼로 변환
            class _NamedFile:
                def __init__(self, path):
                    self._f = open(path, "rb")
                    self.name = path.name
                def read(self, *a): return self._f.read(*a)
                def seek(self, *a): return self._f.seek(*a)
                def tell(self): return self._f.tell()
                def __enter__(self): return self
                def __exit__(self, *a): self._f.close()

            image_files = [_NamedFile(_p) for _p in _auto_paths]
            st.session_state["t2_image_files_names"] = [_p.name for _p in _auto_paths]
        else:
            image_files = st.file_uploader(
                "이미지 파일들", type=["jpg","jpeg","png","bmp","webp"],
                accept_multiple_files=True, key="t2_images"
            )
            if image_files:
                st.session_state["t2_image_files_names"] = [f.name for f in image_files]
            if st.session_state.get("t2_img_manual_mode") and st.button("↩ 자동 이미지로 돌아가기", key="t2_img_auto_back", use_container_width=True):
                st.session_state.pop("t2_img_manual_mode", None)
                st.rerun()

    # ── ③ product_id CSV 업로드
    with col_u3:
        st.markdown("**③ product_id CSV**")
        inherited_csv_df = st.session_state.get("t2_csv_df")

        # 실행 경로의 product_id.csv 자동 탐색
        if inherited_csv_df is None and not st.session_state.get("t2_csv_manual_mode"):
            _csv_path = Path(__file__).parent / "product_id.csv"
            if _csv_path.exists():
                try:
                    _df = pd.read_csv(_csv_path)
                    st.session_state["t2_csv_df"] = _df
                    st.session_state["t2_csv_auto_name"] = _csv_path.name
                    inherited_csv_df = _df
                except Exception as _e:
                    st.warning(f"product_id.csv 자동 로드 실패: {_e}")

        if inherited_csv_df is not None and not st.session_state.get("t2_csv_manual_mode"):
            _auto_name = st.session_state.get("t2_csv_auto_name", "")
            _badge = f"✅ {_auto_name} 자동 로드됨" if _auto_name else f"✅ 기존 CSV 유지 중"
            st.markdown(
                f'<div class="auto-badge">{_badge} ({len(inherited_csv_df):,}행)</div>',
                unsafe_allow_html=True
            )
            if st.button("🔄 CSV 재업로드", key="t2_csv_reset", use_container_width=True):
                st.session_state.pop("t2_csv_df", None)
                st.session_state.pop("t2_csv_auto_name", None)
                st.session_state["t2_csv_manual_mode"] = True
                st.rerun()
            csv_file = None
        else:
            csv_file = st.file_uploader("CSV 파일", type=["csv"], key="t2_csv")
            if csv_file:
                try:
                    _df = pd.read_csv(csv_file)
                    st.session_state["t2_csv_df"] = _df
                    st.session_state.pop("t2_csv_manual_mode", None)
                    st.success(f"✅ CSV {len(_df):,}행 로드")
                except Exception as e:
                    st.error(f"CSV 오류: {e}")
            if st.session_state.get("t2_csv_manual_mode") and st.button("↩ 자동 CSV로 돌아가기", key="t2_csv_auto_back", use_container_width=True):
                st.session_state.pop("t2_csv_manual_mode", None)
                st.rerun()

    # Detection 모드 전용 설정 UI
    if not _t2_is_cls:
        conf_thresh = st.slider("신뢰도 임계값", 0.1, 0.9, 0.25, 0.05, key="t2_conf")
    else:
        conf_thresh = 0.5

    # ── 검사 실행 버튼 및 처리
    if st.button("🚀 검사 실행 & RAG TXT 생성", key="t2_run_btn"):
        if not _use_inherited_model and not model_file:
            st.error("❌ 모델 파일이 없습니다. 비전 모델 생성 탭에서 학습하거나 파일을 업로드하세요.")
            st.stop()
        if not image_files:
            st.error("❌ 이미지 파일을 업로드하세요.")
            st.stop()

        tmp_dir2 = tempfile.mkdtemp()

        # 사용할 모델 파일 경로 결정
        if _use_inherited_model:
            pt_source_path = inherited_model["best_pt"]
        else:
            _ext = os.path.splitext(model_file.name)[1] or ".pt"
            pt_source_path = os.path.join(tmp_dir2, f"model{_ext}")
            with open(pt_source_path, "wb") as f:
                f.write(model_file.read())

        # CSV에서 이미지명 → product_id 매핑 생성
        pid_map = {}
        _csv_df = st.session_state.get("t2_csv_df")
        if _csv_df is not None:
            if "image_filename" in _csv_df.columns and "product_id" in _csv_df.columns:
                for _, row in _csv_df.iterrows():
                    pid_map[str(row["image_filename"]).strip()] = str(row["product_id"]).strip()

        results_list = []
        prog2 = st.progress(0)

        # ── ═
        # Classification 모드 검사 (torchvision .pth)
        # ── ═
        if _t2_is_cls:
            try:
                import torch
                from torchvision import transforms, models as tvm
                from PIL import Image as _PIL2
            except ImportError as _ie:
                st.error(f"❌ pip install torch torchvision pillow ({_ie})")
                st.stop()

            with st.spinner("Classification 모델 로딩 중..."):
                _ckpt    = torch.load(pt_source_path, map_location="cpu")
                _mname   = _ckpt.get("model_name", "ResNet-18")
                _classes = _ckpt.get("classes", ["normal", "defect"])

                if "EfficientNet" in _mname:
                    _mdl = tvm.efficientnet_b0(weights=None)
                    _mdl.classifier[1] = torch.nn.Linear(_mdl.classifier[1].in_features, 2)
                elif "MobileNet" in _mname:
                    _mdl = tvm.mobilenet_v3_small(weights=None)
                    _mdl.classifier[3] = torch.nn.Linear(_mdl.classifier[3].in_features, 2)
                else:
                    _mdl = tvm.resnet18(weights=None)
                    _mdl.fc = torch.nn.Linear(_mdl.fc.in_features, 2)

                _mdl.load_state_dict(_ckpt["state_dict"])
                _mdl.eval()

            st.success(f"✅ {_mname} 로드 완료 | 클래스: {_classes}")

            _tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

            for idx, img_f in enumerate(image_files):
                try:
                    _img = _PIL2.open(img_f).convert("RGB")
                    img_f.seek(0)
                    _x = _tf(_img).unsqueeze(0)
                    with torch.no_grad():
                        _probs = torch.softmax(_mdl(_x), dim=1)[0]
                    _pred_idx  = int(_probs.argmax())
                    _conf      = float(_probs[_pred_idx])
                    _label     = _classes[_pred_idx]
                    _defect    = "NoDefects" if _label == "normal" else "Defect"
                    pid = pid_map.get(img_f.name,
                          pid_map.get(os.path.splitext(img_f.name)[0],
                          os.path.splitext(img_f.name)[0]))
                    results_list.append({
                        "filename":    img_f.name,
                        "product_id":  pid,
                        "defect":      _defect,
                        "confidence":  round(_conf, 4),
                        "detections":  [{"class": _label, "confidence": round(_conf,4)}],
                        "prob_normal": round(float(_probs[0]), 4),
                        "prob_defect": round(float(_probs[1]), 4),
                    })
                except Exception as e:
                    results_list.append({
                        "filename": img_f.name, "product_id": img_f.name,
                        "defect": "Error", "confidence": 0.0,
                        "detections": [], "error": str(e)
                    })
                prog2.progress(int((idx+1)/len(image_files)*100))

        # ── ═
        # Detection 모드 검사 (YOLO .pt/.onnx)
        # ── ═
        else:
            try:
                from ultralytics import YOLO
            except ImportError:
                st.error("❌ pip install ultralytics")
                st.stop()

            img_dir = os.path.join(tmp_dir2, "images")
            os.makedirs(img_dir, exist_ok=True)

            with st.spinner("모델 로딩 중..."):
                yolo_model  = YOLO(pt_source_path)
                class_names = yolo_model.names
            st.success(f"✅ 모델 로드 완료 | 클래스: {list(class_names.values())}")

            for idx, img_f in enumerate(image_files):
                img_path = os.path.join(img_dir, img_f.name)
                with open(img_path, "wb") as f:
                    f.write(img_f.read())
                try:
                    preds = yolo_model(img_path, conf=conf_thresh, verbose=False)
                    pred  = preds[0]
                    detections = []
                    if pred.boxes is not None and len(pred.boxes) > 0:
                        for box in pred.boxes:
                            cls_nm = class_names.get(int(box.cls[0]), str(int(box.cls[0])))
                            detections.append({
                                "class": cls_nm,
                                "confidence": round(float(box.conf[0]), 4),
                                "bbox": [round(v,1) for v in box.xyxy[0].tolist()]
                            })
                    if detections:
                        best_det = max(detections, key=lambda d: d["confidence"])
                        defect   = best_det["class"]
                        top_conf = best_det["confidence"]
                    else:
                        defect   = "NoDefects"
                        top_conf = 1.0
                    pid = pid_map.get(img_f.name,
                          pid_map.get(os.path.splitext(img_f.name)[0],
                          os.path.splitext(img_f.name)[0]))
                    results_list.append({
                        "filename":   img_f.name,
                        "product_id": pid,
                        "defect":     defect,
                        "confidence": top_conf,
                        "detections": detections
                    })
                except Exception as e:
                    results_list.append({
                        "filename": img_f.name, "product_id": img_f.name,
                        "defect": "Error", "confidence": 0.0,
                        "detections": [], "error": str(e)
                    })
                prog2.progress(int((idx+1)/len(image_files)*100))

        st.session_state["t2_results"] = results_list
        shutil.rmtree(tmp_dir2)
        st.success(f"✅ {len(image_files)}개 이미지 검사 완료!")


    if "t2_results" in st.session_state:
        results_list = st.session_state["t2_results"]
        st.markdown('<div class="section-header">📋 검사 결과 요약</div>', unsafe_allow_html=True)

        defect_counts = {}
        for r in results_list:
            defect_counts[r["defect"]] = defect_counts.get(r["defect"], 0) + 1

        stat_cols = st.columns(max(len(defect_counts), 1))
        for i, (defect, cnt) in enumerate(sorted(defect_counts.items())):
            color = DEFECT_COLORS.get(defect, "#888")
            with stat_cols[i]:
                st.markdown(
                    f'<div style="background:#161616;border:2px solid {color};border-radius:6px;'
                    f'padding:14px;text-align:center">'
                    f'<div style="color:{color};font-family:monospace;font-size:1.4rem;font-weight:600">{cnt}</div>'
                    f'<div style="color:#888;font-size:0.8rem;margin-top:4px">{defect}</div></div>',
                    unsafe_allow_html=True
                )

        # Classification 모드: 클래스별 확률 컬럼 추가
        if _t2_is_cls:
            df_show = pd.DataFrame([{
                "파일명":       r["filename"],
                "product_id":   r["product_id"],
                "판정":         r["defect"],
                "신뢰도":       f"{r['confidence']*100:.1f}%",
                "정상 확률":    f"{r.get('prob_normal',0)*100:.1f}%",
                "불량 확률":    f"{r.get('prob_defect',0)*100:.1f}%",
            } for r in results_list])
        else:
            df_show = pd.DataFrame([{
                "파일명":       r["filename"],
                "product_id":   r["product_id"],
                "불량 유형":    r["defect"],
                "신뢰도":       f"{r['confidence']*100:.1f}%",
                "탐지 수":      len(r["detections"])
            } for r in results_list])
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">📄 RAG용 TXT 생성</div>', unsafe_allow_html=True)

        defect_groups = {}
        for r in results_list:
            defect_groups.setdefault(r["defect"], []).append(r)

        txt_lines = [
            "="*60,
            "초콜릿 품질 검사 결과 보고서 (AI 검사)",
            f"생성 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"총 검사 수량: {len(results_list)}개",
            f"모드: {'Classification' if _t2_is_cls else 'Detection (YOLO)'}",
            "="*60, ""
        ]

        # ── 불량률 요약 블록 (LLM이 바로 참조할 수 있도록 명시)
        _total = len(results_list)
        _defect_cnt = {d: len(items) for d, items in defect_groups.items()}
        _n_defect = sum(cnt for d, cnt in _defect_cnt.items() if d not in ("NoDefects", "Error"))
        _n_normal = _defect_cnt.get("NoDefects", 0)
        _rate_defect = _n_defect / _total * 100 if _total else 0
        _rate_normal = _n_normal / _total * 100 if _total else 0

        txt_lines += [
            "[검사 결과 요약]",
            f"  총 검사 수량    : {_total}개",
            f"  정상(NoDefects) : {_n_normal}개  ({_rate_normal:.1f}%)",
            f"  불량 합계       : {_n_defect}개  ({_rate_defect:.1f}%)",
        ]
        for d, cnt in sorted(_defect_cnt.items()):
            if d in ("NoDefects", "Error"): continue
            txt_lines.append(
                f"    └ {d:15s}: {cnt}개  ({cnt/_total*100:.1f}%)"
            )
        txt_lines += ["", "="*60, ""]

        defect_desc = {
            "Bloom":      "블루밍(Bloom) 불량 - 표면 흰색/회색 얼룩",
            "Crack":      "균열(Crack) 불량 - 표면/내부 균열",
            "Crack_Bloom":"복합(Crack_Bloom) 불량 - 균열+블루밍 동시 발생",
            "Defect":     "불량(Defect) - Classification 모델 판정",
            "NoDefects":  "정상(NoDefects) - 불량 없음",
            "Error":      "검사 오류(Error)",
        }
        for defect_type, items in defect_groups.items():
            txt_lines += [
                f"[{defect_desc.get(defect_type, defect_type)}]",
                f"해당 제품 수: {len(items)}개", ""
            ]
            for item in items:
                line = (f"  제품ID: {item['product_id']} | 파일명: {item['filename']} | "
                        f"판정: {item['defect']} | 신뢰도: {item['confidence']*100:.1f}%")
                if _t2_is_cls:
                    line += (f" | 정상확률: {item.get('prob_normal',0)*100:.1f}%"
                             f" | 불량확률: {item.get('prob_defect',0)*100:.1f}%")
                txt_lines.append(line)
            txt_lines += ["-"*40, ""]

        txt_lines += ["[불량 제품 목록 (빠른 참조)]"]
        for r in results_list:
            if r["defect"] not in ("NoDefects",):
                txt_lines.append(
                    f"  product_id={r['product_id']} | 불량={r['defect']} | 신뢰도={r['confidence']*100:.1f}%"
                )
        txt_lines += ["", "="*60]

        rag_txt = "\n".join(txt_lines)

        st.download_button(
            label="⬇ RAG용 TXT 다운로드",
            data=rag_txt.encode("utf-8"),
            file_name="chocolate_inspection_rag.txt",
            mime="text/plain",
            use_container_width=True,
            key="rag_txt_dl"
        )
        st.session_state["generated_rag_txt"] = rag_txt
        st.info("💡 위 버튼을 눌러 검사 결과 TXT 파일을 다운로드하세요. AI 챗봇 질의응답 탭에서 활용할 수 있습니다.")


# ────────────────────────────────────────────────────────────
# TAB: AI 챗봇 질의응답
# ────────────────────────────────────────────────────────────

PROCESS_SENSORS = {
    "로스팅":["temperature_c","pressure_bar","smoke_adc"],
    "분쇄":  ["temperature_c","vibration_hz","particle_size_um"],
    "콘칭":  ["temperature_c","humidity_pct","viscosity_cp"],
    "템퍼링":["temperature_c","viscosity_cp"],
    "몰딩":  ["temperature_c","humidity_pct","pressure_bar"],
    "냉각":  ["temperature_c","humidity_pct","cooling_temp_c"],
}
SENSOR_THRESHOLDS = {
    "temperature_c":{
        "로스팅":{"warn_hi":230,"crit_hi":245,"warn_lo":195,"crit_lo":190,"unit":"°C"},
        "분쇄":  {"warn_hi":65, "crit_hi":70, "warn_lo":40, "crit_lo":35, "unit":"°C"},
        "콘칭":  {"warn_hi":80, "crit_hi":85, "warn_lo":65, "crit_lo":60, "unit":"°C"},
        "템퍼링":{"warn_hi":42, "crit_hi":48, "warn_lo":25, "crit_lo":20, "unit":"°C"},
        "몰딩":  {"warn_hi":32, "crit_hi":35, "warn_lo":28, "crit_lo":26, "unit":"°C"},
        "냉각":  {"warn_hi":22, "crit_hi":25, "warn_lo":16, "crit_lo":14, "unit":"°C"},
    },
    "humidity_pct":    {"warn_hi":60,  "crit_hi":70,  "warn_lo":45,  "crit_lo":40,  "unit":"%"},
    "cooling_temp_c":  {"warn_hi":14,  "crit_hi":16,  "warn_lo":8,   "crit_lo":6,   "unit":"°C"},
    "pressure_bar":    {"warn_hi":1.1, "crit_hi":1.2, "warn_lo":0.7, "crit_lo":0.6, "unit":"bar"},
    "viscosity_cp":    {"warn_hi":8000,"crit_hi":9000,"warn_lo":3000,"crit_lo":2000,"unit":"cP"},
    "vibration_hz":    {"warn_hi":85,  "crit_hi":95,  "warn_lo":50,  "crit_lo":40,  "unit":"Hz"},
    "smoke_adc":       {"warn_hi":1400,"crit_hi":1600,"warn_lo":600, "crit_lo":500, "unit":"ADC"},
    "particle_size_um":{"warn_hi":30,  "crit_hi":35,  "warn_lo":15,  "crit_lo":10,  "unit":"μm"},
}
SENSOR_KR = {
    "temperature_c":"온도","humidity_pct":"습도","pressure_bar":"압력",
    "viscosity_cp":"점도","vibration_hz":"진동","smoke_adc":"연기",
    "cooling_temp_c":"냉각온도","particle_size_um":"입자크기",
}

def get_threshold(sensor, process=None):
    thr = SENSOR_THRESHOLDS.get(sensor, {})
    if isinstance(thr, dict) and process and process in thr: return thr[process]
    elif isinstance(thr, dict) and "warn_hi" in thr: return thr
    return None

def classify_sensor_status(val, thr):
    if thr is None or val is None or (isinstance(val, float) and np.isnan(val)): return "normal"
    if val >= thr.get("crit_hi", float("inf")) or val <= thr.get("crit_lo", float("-inf")): return "crit"
    if val >= thr.get("warn_hi", float("inf")) or val <= thr.get("warn_lo", float("-inf")): return "warn"
    return "normal"

@st.cache_resource(show_spinner=False)
def load_embedder():
    try:
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True), "flagembedding"
    except Exception as e:
        st.warning(f"FlagEmbedding 실패: {e}")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("BAAI/bge-m3"), "sentence_transformers"
    except Exception as e:
        st.warning(f"sentence_transformers 실패: {e}")
    return None, None

def embed_texts(model, backend, texts):
    if backend == "flagembedding":
        out = model.encode(texts, batch_size=8, max_length=512,
                           return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return np.array(out["dense_vecs"])
    elif backend == "sentence_transformers":
        return np.array(model.encode(texts, normalize_embeddings=True))
    return None

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a, b.T)

def chunk_text(text, chunk_size=400, overlap=80):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []; current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current: chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    sub = para[i:i+chunk_size]
                    if sub.strip(): chunks.append(sub)
                current = ""
            else:
                current = para
    if current: chunks.append(current)
    return chunks

def query_ollama(prompt, model_name, ollama_url, temperature=0.3):
    import urllib.request
    system_txt = (
        "당신은 초콜릿 공장 품질 관리 전문가 AI입니다. "
        "제공된 문서 컨텍스트를 바탕으로 정확하고 실용적인 답변을 한국어로 제공하세요. "
        "컨텍스트에 없는 내용은 문서에서 확인할 수 없습니다라고 명시하세요."
    )
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "system": system_txt,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": 2048},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_url}/api/generate", data=payload,
        headers={"Content-Type":"application/json"}, method="POST"
    )
    tokens = []
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        tokens.append(obj["response"])
                    if obj.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return f"[Ollama 오류] {e}\nollama serve 후 ollama pull {model_name} 실행 확인"

    result = "".join(tokens).strip()
    if not result:
        return "[응답 없음] 모델이 빈 답변을 반환했습니다. ollama serve 실행 여부와 모델명을 확인하세요."
    return result

with tab_chat:
    st.markdown("""<div style="background:#0d1520;border-left:4px solid #60c0f0;padding:18px 24px;border-radius:4px;margin-bottom:20px">
      <h3 style="color:#60c0f0;font-family:'IBM Plex Mono',monospace;margin:0 0 6px 0">💬 AI 챗봇 질의응답</h3>
      <p style="color:#666;margin:0;font-size:0.85rem">검사 결과와 제품 매뉴얼을 바탕으로 불량 원인, 조치 방법 등을 AI에게 바로 물어볼 수 있습니다.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📂 데이터 소스 확인 & 업로드")
    col_u1, col_u2, col_u3 = st.columns(3)

    # ── ① 검사 결과 TXT: 비전 불량 검사 탭 자동 연결
    with col_u1:
        st.markdown("**① 검사 결과 TXT**")
        inherited_rag_txt = st.session_state.get("generated_rag_txt")

        if inherited_rag_txt:
            st.markdown(
                '<div class="auto-badge">✅ 비전 불량 검사 탭 결과 자동 연결됨</div>',
                unsafe_allow_html=True
            )
            rag_text_content = inherited_rag_txt
            if st.button("🔄 다른 TXT 업로드", key="t3_rag_reset", use_container_width=True):
                del st.session_state["generated_rag_txt"]
                st.rerun()
        else:
            rag_upload = st.file_uploader("검사 결과 TXT", type=["txt"], key="t3_rag")
            if rag_upload:
                rag_text_content = rag_upload.read().decode("utf-8", errors="replace")
                st.session_state["generated_rag_txt"] = rag_text_content
            else:
                rag_text_content = None

    # ── ② 매뉴얼 TXT: 실행 경로의 manual.txt 자동 로드
    if "t3_manual_text" not in st.session_state and not st.session_state.get("t3_manual_manual_mode"):
        _script_dir = Path(__file__).parent
        _manual_path = _script_dir / "manual.txt"
        try:
            with open(_manual_path, encoding="utf-8") as _mf:
                st.session_state["t3_manual_text"] = _mf.read()
                st.session_state["t3_manual_name"] = _manual_path.name
        except Exception as _e:
            st.session_state["t3_manual_text"] = None
            st.session_state["t3_manual_name"] = None

    with col_u2:
        st.markdown("**② 매뉴얼 TXT**")
        inherited_manual = st.session_state.get("t3_manual_text")

        if inherited_manual and not st.session_state.get("t3_manual_manual_mode"):
            manual_name = st.session_state.get("t3_manual_name", "매뉴얼")
            st.markdown(
                f'<div class="auto-badge">✅ {manual_name} 로드됨</div>',
                unsafe_allow_html=True
            )
            manual_text_content = inherited_manual
            if st.button("🔄 다른 매뉴얼 업로드", key="t3_manual_reset", use_container_width=True):
                st.session_state.pop("t3_manual_text", None)
                st.session_state.pop("t3_manual_name", None)
                st.session_state["t3_manual_manual_mode"] = True
                st.rerun()
        else:
            manual_upload = st.file_uploader("매뉴얼 TXT", type=["txt"], key="t3_manual")
            if manual_upload:
                manual_text_content = manual_upload.read().decode("utf-8", errors="replace")
                st.session_state["t3_manual_text"] = manual_text_content
                st.session_state["t3_manual_name"] = manual_upload.name
                st.session_state.pop("t3_manual_manual_mode", None)
                st.rerun()
            else:
                manual_text_content = None
            if st.session_state.get("t3_manual_manual_mode") and st.button("↩ 자동 매뉴얼로 돌아가기", key="t3_manual_auto_back", use_container_width=True):
                st.session_state.pop("t3_manual_manual_mode", None)
                st.rerun()

    # ── ③ 센서 CSV: 검사 탭 CSV 자동 연결 또는 sensor_data.csv 자동 탐색
    with col_u3:
        st.markdown("**③ 센서 데이터 CSV**")
        sensor_df = None

        # ── sensor_data.csv 자동 로드 (세션 유지)
        if "t3_sensor_df" not in st.session_state and not st.session_state.get("t3_sensor_manual_mode"):
            _sensor_path = Path(__file__).parent / "sensor_data.csv"
            if _sensor_path.exists():
                try:
                    _sdf = pd.read_csv(_sensor_path)
                    req_cols = {"product_id","defect_label","process","timestamp"}
                    if req_cols.issubset(_sdf.columns):
                        st.session_state["t3_sensor_df"] = _sdf
                        st.session_state["t3_sensor_auto_name"] = _sensor_path.name
                    else:
                        st.warning(f"sensor_data.csv 필수 컬럼 누락: {req_cols - set(_sdf.columns)}")
                except Exception as _e:
                    st.warning(f"sensor_data.csv 자동 로드 실패: {_e}")

        # 세션에서 sensor_df 불러오기
        sensor_df = st.session_state.get("t3_sensor_df")

        if sensor_df is not None and not st.session_state.get("t3_sensor_manual_mode"):
            _auto_name = st.session_state.get("t3_sensor_auto_name", "")
            _badge = f"✅ {_auto_name} 자동 로드됨" if _auto_name else "✅ 센서 CSV 로드됨"
            st.markdown(
                f'<div class="auto-badge">{_badge} ({len(sensor_df):,}행)</div>',
                unsafe_allow_html=True
            )
            if st.button("🔄 다른 센서 CSV 업로드", key="t3_sensor_reset", use_container_width=True):
                st.session_state.pop("t3_sensor_df", None)
                st.session_state.pop("t3_sensor_auto_name", None)
                st.session_state["t3_sensor_manual_mode"] = True
                st.rerun()
        else:
            sensor_upload = st.file_uploader("센서 CSV", type=["csv"], key="t3_sensor")
            if sensor_upload:
                try:
                    _sdf = pd.read_csv(sensor_upload)
                    req_cols = {"product_id","defect_label","process","timestamp"}
                    if not req_cols.issubset(_sdf.columns):
                        st.error(f"❌ 필수 컬럼 누락: {req_cols - set(_sdf.columns)}")
                    else:
                        st.session_state["t3_sensor_df"] = _sdf
                        st.session_state.pop("t3_sensor_manual_mode", None)
                        sensor_df = _sdf
                        st.success(f"✅ 센서 데이터 {len(_sdf):,}행 로드")
                except Exception as e:
                    st.error(f"CSV 오류: {e}")
            if st.session_state.get("t3_sensor_manual_mode") and st.button("↩ 자동 CSV로 돌아가기", key="t3_sensor_auto_back", use_container_width=True):
                st.session_state.pop("t3_sensor_manual_mode", None)
                st.rerun()

    with st.expander("⚙️ LLM / RAG 설정", expanded=False):
        t3_ollama_url  = st.text_input("Ollama URL",   value="http://localhost:11434", key="t3_url")
        t3_llm_model   = st.text_input("LLM 모델명",  value="qwen3:8b",               key="t3_llm")
        t3_top_k       = st.slider("top-k 청크",      1, 10, 5,     key="t3_topk")
        t3_temperature = st.slider("Temperature",      0.0, 1.0, 0.3, 0.05, key="t3_temp")
        t3_chunk_size  = st.slider("청크 크기 (문자)", 100, 800, 400, 50,   key="t3_chunk")

    can_index = rag_text_content or manual_text_content or (sensor_df is not None)

    if can_index and st.button("🔧 임베딩 인덱스 빌드 (BGE-M3)", key="t3_index_btn"):
        with st.spinner("BGE-M3 모델 로딩 중... (최초 1회만 다운로드)"):
            embedder, backend = load_embedder()
        if embedder is None:
            st.error("❌ pip install FlagEmbedding 또는 pip install sentence-transformers")
        else:
            all_chunks = []; sources = []
            if rag_text_content:
                c = chunk_text(rag_text_content, chunk_size=t3_chunk_size)
                all_chunks.extend(c); sources.extend(["검사결과"]*len(c))
            if manual_text_content:
                c = chunk_text(manual_text_content, chunk_size=t3_chunk_size)
                all_chunks.extend(c); sources.extend(["매뉴얼"]*len(c))
            if sensor_df is not None:
                DEFECT_LABEL_KR = {
                    "Bloom":       "Bloom(블루밍 단독 — 균열 없음)",
                    "Crack":       "Crack(균열 단독 — 블루밍 없음)",
                    "Crack_Bloom": "Crack_Bloom(균열+블루밍 복합 — Bloom·Crack과 별개 클래스)",
                    "NoDefects":   "NoDefects(정상 — 불량 없음)",
                }
                for proc, sensors_p in PROCESS_SENSORS.items():
                    for defect in ["Bloom","Crack","Crack_Bloom","NoDefects"]:
                        sub = sensor_df[(sensor_df["process"]==proc)&(sensor_df["defect_label"]==defect)]
                        if sub.empty: continue
                        defect_full = DEFECT_LABEL_KR.get(defect, defect)
                        lines_s = [f"[{proc} 공정 / {defect_full}]", f"레코드: {len(sub)}건"]
                        for s in sensors_p:
                            if s not in sub.columns: continue
                            col_d = sub[s].dropna()
                            if col_d.empty: continue
                            thr  = get_threshold(s, proc)
                            unit = thr.get("unit","") if thr else ""
                            mv   = col_d.mean()
                            st_  = classify_sensor_status(mv, thr)
                            flag = {"normal":"✅","warn":"⚠️","crit":"🚨"}[st_]
                            lines_s.append(f"  {SENSOR_KR.get(s,s)}: 평균={mv:.2f}{unit} {flag}, 최소={col_d.min():.2f}{unit}, 최대={col_d.max():.2f}{unit}")
                        pids = sub["product_id"].unique()
                        lines_s.append(f"  제품ID: {', '.join(pids[:20])}" + (" ..." if len(pids)>20 else ""))
                        all_chunks.append("\n".join(lines_s)); sources.append("센서데이터")
            if not all_chunks:
                st.error("생성된 청크가 없습니다.")
            else:
                with st.spinner(f"{len(all_chunks)}개 청크 임베딩 중..."):
                    vecs = embed_texts(embedder, backend, all_chunks)
                st.session_state["t3_chunks"]   = all_chunks
                st.session_state["t3_sources"]  = sources
                st.session_state["t3_vecs"]     = vecs
                st.session_state["t3_embedder"] = embedder
                st.session_state["t3_backend"]  = backend
                counts = {s: sources.count(s) for s in set(sources)}
                st.success(f"✅ 인덱스 완료 | {len(all_chunks)}개 청크 | " + " / ".join(f"{k}:{v}" for k,v in counts.items()))

    if "t3_chunks" in st.session_state:
        n_c = len(st.session_state["t3_chunks"])
        src_c = {s: st.session_state["t3_sources"].count(s) for s in set(st.session_state["t3_sources"])}
        st.markdown(
            f'<div style="background:#0d1a0d;border:1px solid #4caf50;border-radius:4px;padding:8px 16px;color:#81c784;font-family:monospace;font-size:0.82rem;margin:8px 0">'
            f'✅ 인덱스 준비됨 — {n_c}개 | ' + " / ".join(f"{k}:{v}" for k,v in src_c.items()) + '</div>',
            unsafe_allow_html=True
        )

    # ── 챗봇 session_state 초기화
    if "t3_history" not in st.session_state:
        st.session_state["t3_history"] = []
    if "t3_quick_q" not in st.session_state:
        st.session_state["t3_quick_q"] = None

    st.markdown("### 💬 질의응답")

    # ── 추천 질문 버튼
    QUICK_QUESTIONS = [
        "전체 불량률 요약해줘",
        "Crack 불량 제품 목록 알려줘",
        "Bloom 불량 제품 목록 알려줘",
        "Crack_Bloom 복합 불량 처리 방법은?",
        "균열 불량 원인과 조치 방법은?",
        "블루밍 예방 조건은?",
        "온도·습도 경고 기준 알려줘",
        "Crack_Bloom 불량 원인은?",
        "정기 점검 항목 알려줘",
    ]
    st.markdown("**💡 추천 질문**")
    btn_cols = st.columns(3)
    for i, q in enumerate(QUICK_QUESTIONS):
        if btn_cols[i % 3].button(q, key=f"t3_qq_{i}", use_container_width=True):
            st.session_state["t3_quick_q"] = q

    st.divider()

    # ── 대화 기록 렌더링
    for msg in st.session_state["t3_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.write(msg["content"])
            else:
                st.write(msg["content"])

    # ── 사용자 입력 처리 (직접 입력 또는 추천 버튼)
    user_input = st.chat_input("질문을 입력하세요...",
                               key="t3_chat_input",
                               disabled=("t3_chunks" not in st.session_state))

    # 추천 버튼 클릭 시 해당 질문을 입력으로 사용
    if st.session_state["t3_quick_q"]:
        user_input = st.session_state["t3_quick_q"]
        st.session_state["t3_quick_q"] = None

    if user_input:
        if "t3_chunks" not in st.session_state:
            st.warning("⚠️ 먼저 데이터를 업로드하고 임베딩 인덱스 빌드 버튼을 클릭하세요.")
        else:
            query    = user_input.strip()
            embedder = st.session_state["t3_embedder"]
            backend  = st.session_state["t3_backend"]
            chunks   = st.session_state["t3_chunks"]
            sources  = st.session_state["t3_sources"]
            vecs     = st.session_state["t3_vecs"]

            # 사용자 메시지 화면에 즉시 표시
            with st.chat_message("user"):
                st.write(query)
            st.session_state["t3_history"].append({"role": "user", "content": query})

            # BGE-M3 임베딩으로 관련 문서 검색
            with st.spinner("관련 문서 검색 중 (BGE-M3)..."):
                q_vec   = embed_texts(embedder, backend, [query])
                sims    = cosine_sim(q_vec, vecs)[0]
                top_idx = np.argsort(sims)[::-1][:t3_top_k]

            context_parts = [
                f"[참고{rank} | {sources[idx]} | 유사도:{sims[idx]:.3f}]\n{chunks[idx]}"
                for rank, idx in enumerate(top_idx, 1)
            ]
            context     = "\n\n---\n\n".join(context_parts)
            history_str = "".join(
                f"{'사용자' if m['role']=='user' else 'AI'}: {m['content'][:200]}\n"
                for m in st.session_state["t3_history"][-8:]
            )

            full_prompt = (
                "아래 컨텍스트를 참고하여 질문에 상세히 답하세요.\n"
                "수치 언급 시 반드시 단위를 포함하세요.\n"
                "마크다운 없이 일반 텍스트로, 한국어로만 답변하세요.\n\n"
                "【중요 — 불량률 계산 규칙】\n"
                "불량률(%) = 불량 합계 ÷ 총 검사 수량 × 100. NoDefects(정상)는 불량에 포함하지 않습니다.\n"
                "컨텍스트의 수치를 직접 계산하여 결과만 답하세요. 계산 과정·출처 문장은 답변에 쓰지 마세요.\n\n"
                "【중요 — 불량 유형 구분 규칙】\n"
                "아래 4가지 클래스는 완전히 독립된 별개의 분류입니다. 절대로 혼동하거나 중복 포함하지 마세요.\n"
                "  · Bloom(블루밍) : 표면 흰색·회색 얼룩만 발생. 균열 없음.\n"
                "  · Crack(균열) : 표면·내부 균열만 발생. 블루밍 없음.\n"
                "  · Crack_Bloom(복합) : 균열과 블루밍이 동시에 발생한 경우. Bloom도 Crack도 아닌 별도 클래스.\n"
                "  · NoDefects(정상) : 아무 불량 없음.\n"
                "Bloom 질문에 Crack_Bloom을 포함하거나, Crack 질문에 Crack_Bloom을 포함하는 것은 오답입니다.\n"
                "각 클래스별로 명확히 분리하여 답변하세요.\n\n"
                "=== 컨텍스트 ===\n" + context + "\n\n"
                "=== 대화 히스토리 ===\n" + history_str +
                "=== 현재 질문 ===\n" + query + "\n\n답변:"
            )

            # 검색된 문서 청크 미리보기
            source_colors = {"검사결과": "#60c0f0", "매뉴얼": "#f0a060", "센서데이터": "#81c784"}
            with st.expander(f"🔍 검색된 참고 문서 (top-{t3_top_k})", expanded=False):
                for rank, idx in enumerate(top_idx, 1):
                    color = source_colors.get(sources[idx], "#888")
                    st.markdown(
                        f'<div style="background:#161616;border-left:3px solid {color};padding:10px;'
                        f'margin-bottom:6px;border-radius:3px;font-size:0.8rem">'
                        f'<b style="color:{color}">[{rank}] {sources[idx]} | 유사도: {sims[idx]:.3f}</b><br>'
                        f'<span style="color:#aaa">{chunks[idx][:300]}…</span></div>',
                        unsafe_allow_html=True
                    )

            # LLM 답변 생성 및 스트리밍 출력
            with st.chat_message("assistant"):
                with st.spinner(f"{t3_llm_model} 답변 생성 중..."):
                    answer = query_ollama(full_prompt, model_name=t3_llm_model,
                                         ollama_url=t3_ollama_url, temperature=t3_temperature)
                # thinking / 답변 분리 출력
                import re as _re
                _think_match = _re.search(r"<think>(.*?)</think>", answer, _re.DOTALL)
                _think_txt   = _think_match.group(1).strip() if _think_match else ""
                _answer_txt  = _re.sub(r"<think>.*?</think>", "", answer, flags=_re.DOTALL).strip()

                if _think_txt:
                    with st.expander("🧠 thinking (추론 과정)", expanded=False):
                        st.text(_think_txt)
                st.write(_answer_txt if _answer_txt else answer)

            st.session_state["t3_history"].append({"role": "assistant", "content": _answer_txt if _answer_txt else answer})
            st.rerun()

    # ── 대화 기록 초기화 버튼
    if st.session_state["t3_history"]:
        st.divider()
        if st.button("🗑️ 대화 초기화", key="t3_clear_btn", type="secondary"):
            st.session_state["t3_history"] = []
            st.rerun()



# ────────────────────────────────────────────────────────────
# TAB: 공정 품질 분석 보고서
# ────────────────────────────────────────────────────────────

with tab_report:
    st.markdown("""<div style="background:#1a100a;border-left:4px solid #e05050;padding:18px 24px;border-radius:4px;margin-bottom:24px">
      <h3 style="color:#f08080;font-family:'IBM Plex Mono',monospace;margin:0 0 6px 0">📄 비전 품질 분석 보고서</h3>
      <p style="color:#666;margin:0;font-size:0.85rem">비전 검사 결과 · 센서 데이터 · 분석 내용을 자동으로 모아 PDF로 만들어 드립니다.</p>
    </div>""", unsafe_allow_html=True)

    # ── 연동 데이터 수집
    _rp_vision   = st.session_state.get("t2_results")       # 비전 검사 결과 리스트
    _rp_sensor   = st.session_state.get("t3_sensor_df") if st.session_state.get("t3_sensor_df") is not None else st.session_state.get("t2_csv_df")  # 센서 CSV
    _rp_chat     = st.session_state.get("t3_history", [])   # 챗봇 대화 기록
    _rp_rag_txt  = st.session_state.get("generated_rag_txt")

    # 챗봇 마지막 AI 답변 추출
    _rp_last_ai = ""
    for _m in reversed(_rp_chat):
        if _m["role"] == "assistant":
            _rp_last_ai = _m["content"]
            break

    # ── STEP 1 : 연동 현황 카드
    st.markdown('<div class="section-header">① 연동된 데이터 확인</div>', unsafe_allow_html=True)

    _s1, _s2 = st.columns(2)
    # 비전 검사 카드
    with _s1:
        if _rp_vision:
            _v_total  = len(_rp_vision)
            _v_defect = sum(1 for r in _rp_vision if r["defect"] not in ("NoDefects", "Error"))
            _v_rate   = _v_defect / _v_total * 100 if _v_total else 0
            st.markdown(
                f'<div style="background:#0d1a0d;border:1px solid #4caf50;border-radius:8px;padding:16px 14px">'
                f'<div style="color:#4caf50;font-family:monospace;font-size:0.75rem;margin-bottom:8px">✅ 비전 검사 결과 연결됨</div>'
                f'<div style="display:flex;gap:16px">'
                f'<div><div style="color:#ccc;font-family:monospace;font-size:1.4rem;font-weight:600">{_v_total}</div><div style="color:#666;font-size:0.75rem">총 검사</div></div>'
                f'<div><div style="color:#f06060;font-family:monospace;font-size:1.4rem;font-weight:600">{_v_defect}</div><div style="color:#666;font-size:0.75rem">불량</div></div>'
                f'<div><div style="color:#f0a060;font-family:monospace;font-size:1.4rem;font-weight:600">{_v_rate:.1f}%</div><div style="color:#666;font-size:0.75rem">불량률</div></div>'
                f'</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#161616;border:1px solid #333;border-radius:8px;padding:16px 14px">'
                '<div style="color:#555;font-family:monospace;font-size:0.75rem;margin-bottom:6px">⬜ 비전 검사 미실행</div>'
                '<div style="color:#444;font-size:0.8rem">🔍 비전 모델로 불량 검사 탭에서<br>검사를 실행하면 자동 연결됩니다</div>'
                '</div>', unsafe_allow_html=True)
    # 센서 카드
    with _s2:
        if _rp_sensor is not None:
            _sen_rows = len(_rp_sensor)
            _sen_prods = _rp_sensor["product_id"].nunique() if "product_id" in _rp_sensor.columns else "-"
            st.markdown(
                f'<div style="background:#0d1520;border:1px solid #60c0f0;border-radius:8px;padding:16px 14px">'
                f'<div style="color:#60c0f0;font-family:monospace;font-size:0.75rem;margin-bottom:8px">✅ 센서 데이터 연결됨</div>'
                f'<div style="display:flex;gap:16px">'
                f'<div><div style="color:#ccc;font-family:monospace;font-size:1.4rem;font-weight:600">{_sen_prods}</div><div style="color:#666;font-size:0.75rem">제품 수</div></div>'
                f'<div><div style="color:#ccc;font-family:monospace;font-size:1.4rem;font-weight:600">{_sen_rows:,}</div><div style="color:#666;font-size:0.75rem">레코드</div></div>'
                f'</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#161616;border:1px solid #333;border-radius:8px;padding:16px 14px">'
                '<div style="color:#555;font-family:monospace;font-size:0.75rem;margin-bottom:6px">⬜ 센서 데이터 없음</div>'
                '<div style="color:#444;font-size:0.8rem">📊 AI 챗봇 탭에서 센서 CSV를<br>업로드하면 자동 연결됩니다</div>'
                '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 2 : 보고서 생성 버튼
    st.markdown('<div class="section-header">② PDF 보고서 생성</div>', unsafe_allow_html=True)

    # 최소 데이터 없으면 경고
    if not _rp_vision and _rp_sensor is None:
        st.warning("⚠️ 비전 검사 결과 또는 센서 데이터 중 하나 이상이 있어야 보고서를 생성할 수 있습니다.")

    _gen_disabled = (not _rp_vision and _rp_sensor is None)

    if st.button("📋 PDF 보고서 생성하기", key="t4_gen_btn",
                 use_container_width=True, disabled=_gen_disabled):
        try:
            import io as _io
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.lib import colors as _rl_colors
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer,
                Table, TableStyle, HRFlowable, Image as RLImage, PageBreak
            )
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _rplt
            import matplotlib.font_manager as _rfm

            # ── 한글 폰트 등록
            _rf_name = "Helvetica"
            for _fp in ["C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                        "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
                if os.path.exists(_fp):
                    try:
                        pdfmetrics.registerFont(TTFont("KR", _fp))
                        _rf_name = "KR"
                        _rfm.fontManager.addfont(_fp)
                        _rplt.rcParams["font.family"] = _rfm.FontProperties(fname=_fp).get_name()
                        _rplt.rcParams["axes.unicode_minus"] = False
                    except Exception:
                        pass
                    break

            _rfb = _io.BytesIO()
            _doc = SimpleDocTemplate(_rfb, pagesize=A4,
                                     leftMargin=2.2*cm, rightMargin=2.2*cm,
                                     topMargin=2.2*cm, bottomMargin=2.2*cm)
            _W = A4[0] - 4.4*cm

            # ── 스타일 헬퍼
            def _rs(name, size=10, leading=14, color="#333333",
                    bold=False, align="LEFT", sb=0, sa=4):
                return ParagraphStyle(
                    name, fontName=_rf_name, fontSize=size, leading=leading,
                    textColor=_rl_colors.HexColor(color),
                    alignment={"LEFT":0,"CENTER":1,"RIGHT":2}[align],
                    spaceBefore=sb, spaceAfter=sa, wordWrap="CJK"
                )

            S_TITLE   = _rs("RT",  size=20, color="#1a0a00", align="LEFT", sa=4)
            S_SUB     = _rs("RS",  size=8,  color="#999999", sa=2)
            S_H1      = _rs("RH1", size=13, color="#d4612a", sb=18, sa=6)
            S_H2      = _rs("RH2", size=10, color="#333333", sb=10, sa=4)
            S_BODY    = _rs("RB",  size=9,  color="#333333", leading=14, sa=3)
            S_BADGE   = _rs("RBG", size=8,  color="#888888", sa=2)
            S_CAP     = _rs("RC",  size=7.5, color="#888888", align="CENTER", sa=2)

            _story = []

            # ── 표지
            _story.append(Paragraph("초콜릿 공장 품질 분석 보고서", S_TITLE))
            _story.append(Spacer(1, 3))
            _story.append(Paragraph(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}", S_SUB))
            if _rp_sensor is not None and "timestamp" in _rp_sensor.columns:
                try:
                    _rts = pd.to_datetime(_rp_sensor["timestamp"])
                    _story.append(Paragraph(
                        f"분석 기간: {_rts.min().strftime('%Y-%m-%d')} ~ {_rts.max().strftime('%Y-%m-%d')}", S_SUB))
                except Exception:
                    pass
            _story.append(Spacer(1, 4))
            _story.append(HRFlowable(width=_W, thickness=1.5,
                                     color=_rl_colors.HexColor("#d4612a"), spaceAfter=14))

            # ── 섹션 1 : 불량 현황 요약
            _story.append(Paragraph("1. 불량 현황 요약", S_H1))

            if _rp_vision:
                _rv_total  = len(_rp_vision)
                _rv_defect = sum(1 for r in _rp_vision if r["defect"] not in ("NoDefects","Error"))
                _rv_normal = sum(1 for r in _rp_vision if r["defect"] == "NoDefects")
                _rv_rate   = _rv_defect / _rv_total * 100 if _rv_total else 0

                # 요약 표
                _sum_data = [
                    ["항목", "수량", "비율"],
                    ["총 검사", str(_rv_total), "100%"],
                    ["불량", str(_rv_defect), f"{_rv_rate:.1f}%"],
                    ["정상", str(_rv_normal), f"{100-_rv_rate:.1f}%"],
                ]
                _sum_tbl = Table(_sum_data, colWidths=[_W*0.4, _W*0.3, _W*0.3])
                _sum_tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0),  _rl_colors.HexColor("#d4612a")),
                    ("TEXTCOLOR",     (0,0), (-1,0),  _rl_colors.white),
                    ("FONTNAME",      (0,0), (-1,-1), _rf_name),
                    ("FONTSIZE",      (0,0), (-1,-1), 9),
                    ("ALIGN",         (1,0), (-1,-1), "CENTER"),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1),
                     [_rl_colors.HexColor("#fafafa"), _rl_colors.HexColor("#f0f0f0")]),
                    ("GRID",          (0,0), (-1,-1), 0.3, _rl_colors.HexColor("#cccccc")),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ]))
                _story.append(_sum_tbl)
                _story.append(Spacer(1, 8))

                # 불량 유형 분포 도넛 차트
                _dc_counts = {}
                for _r in _rp_vision:
                    _dc_counts[_r["defect"]] = _dc_counts.get(_r["defect"], 0) + 1

                _DC_COLORS_PDF = {
                    "Bloom":"#f0e060","Crack":"#f06060",
                    "Crack_Bloom":"#f0a060","NoDefects":"#81c784",
                    "Defect":"#e07070","Error":"#888888"
                }
                _fig_d, _ax_d = _rplt.subplots(figsize=(5, 3.2))
                _fig_d.patch.set_facecolor("white"); _ax_d.set_facecolor("white")
                _pie_labels = list(_dc_counts.keys())
                _pie_vals   = list(_dc_counts.values())
                _pie_colors = [_DC_COLORS_PDF.get(l, "#888") for l in _pie_labels]
                _wedges, _texts, _autos = _ax_d.pie(
                    _pie_vals, labels=_pie_labels, autopct="%1.0f%%",
                    colors=_pie_colors, wedgeprops=dict(width=0.5), startangle=90,
                    textprops={"color":"#333","fontsize":8}
                )
                for _at in _autos: _at.set_fontsize(7); _at.set_color("#333")
                _ax_d.set_title("불량 유형 분포", fontsize=10, color="#333", pad=8)
                _fig_d.tight_layout()
                _pie_buf = _io.BytesIO()
                _fig_d.savefig(_pie_buf, format="png", dpi=130,
                               bbox_inches="tight", facecolor="white")
                _rplt.close(_fig_d); _pie_buf.seek(0)
                _story.append(RLImage(_pie_buf, width=_W*0.55, height=_W*0.36))
                _story.append(Spacer(1, 6))

                # 불량 유형별 상세 표
                _story.append(Paragraph("불량 유형별 건수", S_H2))
                _type_data = [["불량 유형", "건수", "비율"]]
                for _lbl, _cnt in sorted(_dc_counts.items()):
                    _type_data.append([_lbl, str(_cnt), f"{_cnt/_rv_total*100:.1f}%"])
                _type_tbl = Table(_type_data, colWidths=[_W*0.45, _W*0.25, _W*0.3])
                _type_tbl.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0),  _rl_colors.HexColor("#555555")),
                    ("TEXTCOLOR",     (0,0), (-1,0),  _rl_colors.white),
                    ("FONTNAME",      (0,0), (-1,-1), _rf_name),
                    ("FONTSIZE",      (0,0), (-1,-1), 8.5),
                    ("ALIGN",         (1,0), (-1,-1), "CENTER"),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1),
                     [_rl_colors.HexColor("#fafafa"), _rl_colors.HexColor("#f0f0f0")]),
                    ("GRID",          (0,0), (-1,-1), 0.3, _rl_colors.HexColor("#cccccc")),
                    ("TOPPADDING",    (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ]))
                _story.append(_type_tbl)
                _story.append(Spacer(1, 6))
            else:
                _story.append(Paragraph("비전 검사 결과 없음 (비전 모델로 불량 검사 탭 미실행)", S_BODY))

            # ── 섹션 2 : 센서 데이터 분석
            if _rp_sensor is not None:
                _story.append(Paragraph("2. 불량 유형별 공정·센서 영향 분석", S_H1))
                _rp_sens_cols = ["temperature_c","humidity_pct","pressure_bar",
                                 "viscosity_cp","vibration_hz","smoke_adc",
                                 "cooling_temp_c","particle_size_um"]
                _rp_sens_kr   = {
                    "temperature_c":"온도(°C)","humidity_pct":"습도(%)",
                    "pressure_bar":"압력(bar)","viscosity_cp":"점도(cP)",
                    "vibration_hz":"진동(Hz)","smoke_adc":"연기(ADC)",
                    "cooling_temp_c":"냉각온도(°C)","particle_size_um":"입자크기(μm)"
                }
                _rp_proc_order = ["로스팅","분쇄","콘칭","템퍼링","몰딩","냉각"]
                _avail_s = [c for c in _rp_sens_cols if c in _rp_sensor.columns]
                _rp_defect_types = ["Bloom","Crack","Crack_Bloom"]
                _rp_dlbl_kr = {"Bloom":"블루밍","Crack":"균열","Crack_Bloom":"균열+블루밍"}
                _rp_dc_hex  = {"Bloom":"#b09000","Crack":"#b02020","Crack_Bloom":"#a05000"}

                if _avail_s and "defect_label" in _rp_sensor.columns and "process" in _rp_sensor.columns:
                    for _dlbl in _rp_defect_types:
                        if _dlbl not in _rp_sensor["defect_label"].values:
                            continue
                        _dc_h = _rp_dc_hex.get(_dlbl, "#555555")
                        _kr   = _rp_dlbl_kr.get(_dlbl, _dlbl)
                        _gmask = _rp_sensor["defect_label"] == _dlbl
                        # 행 수가 아닌 제품 수로 계산 (product_id 기준)
                        if "product_id" in _rp_sensor.columns:
                            _gn = int(_rp_sensor.loc[_gmask, "product_id"].nunique())
                        else:
                            _gn = int(_gmask.sum())

                        # 공정×센서 평균: 불량 그룹 vs 나머지
                        _g_df  = _rp_sensor[_gmask]
                        _o_df  = _rp_sensor[~_gmask]

                        # 공정별 영향도 (불량-정상 평균차 절댓값 합산)
                        _proc_score = {}
                        _proc_top   = {}  # 공정별 top 센서
                        for _pn in _rp_proc_order:
                            _gp = _g_df[_g_df["process"]==_pn][_avail_s]
                            _op = _o_df[_o_df["process"]==_pn][_avail_s]
                            if _gp.empty or _op.empty:
                                continue
                            _diff_p = (_gp.mean() - _op.mean()).abs()
                            # 정규화 (센서 표준편차로 나눔)
                            _std_p  = _rp_sensor[_rp_sensor["process"]==_pn][_avail_s].std().replace(0, 1e-8)
                            _norm_p = _diff_p / _std_p
                            _score_p = float(_norm_p.sum())
                            if _score_p > 0:
                                _proc_score[_pn] = _score_p
                                _top_s = _norm_p.idxmax()
                                _proc_top[_pn] = _top_s

                        if not _proc_score:
                            continue

                        _proc_rank = sorted(_proc_score.items(), key=lambda x: x[1], reverse=True)
                        _max_ps    = _proc_rank[0][1] if _proc_rank else 1

                        # ── 불량 유형 헤더
                        _story.append(Spacer(1, 8))
                        _story.append(Paragraph(
                            f"▶  {_kr}  ({_dlbl})  —  {_gn}개 제품",
                            ParagraphStyle("RPH", fontName=_rf_name, fontSize=10,
                                           textColor=_rl_colors.HexColor(_dc_h),
                                           spaceBefore=6, spaceAfter=4, wordWrap="CJK")
                        ))

                        # ── 공정 영향도 순위 표
                        _rk_data = [["순위", "공정", "영향도", "주요 센서"]]
                        for _ri, (_pn, _pv) in enumerate(_proc_rank[:3]):
                            _bar = "■" * max(1, int(_pv / _max_ps * 10))
                            _top_kr = _rp_sens_kr.get(_proc_top.get(_pn,""), _proc_top.get(_pn,""))
                            _rk_data.append([f"{_ri+1}위", _pn, _bar, _top_kr])
                        _rk_tbl = Table(_rk_data, colWidths=[_W*0.1, _W*0.2, _W*0.45, _W*0.25])
                        _rk_tbl.setStyle(TableStyle([
                            ("BACKGROUND",    (0,0), (-1,0),  _rl_colors.HexColor("#444444")),
                            ("TEXTCOLOR",     (0,0), (-1,0),  _rl_colors.white),
                            ("TEXTCOLOR",     (2,1), (2,-1),  _rl_colors.HexColor(_dc_h)),
                            ("FONTNAME",      (0,0), (-1,-1), _rf_name),
                            ("FONTSIZE",      (0,0), (-1,-1), 8),
                            ("ALIGN",         (0,0), (1,-1),  "CENTER"),
                            ("ROWBACKGROUNDS",(0,1), (-1,-1),
                             [_rl_colors.HexColor("#fafafa"), _rl_colors.HexColor("#f3f3f3")]),
                            ("GRID",          (0,0), (-1,-1), 0.3, _rl_colors.HexColor("#cccccc")),
                            ("TOPPADDING",    (0,0), (-1,-1), 3),
                            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                        ]))
                        _story.append(_rk_tbl)
                        _story.append(Spacer(1, 6))

                        # ── Top 5 센서 영향 표
                        _story.append(Paragraph("주요 원인 센서 Top 5",
                            ParagraphStyle("RPSH", fontName=_rf_name, fontSize=8.5,
                                           textColor=_rl_colors.HexColor("#555555"),
                                           spaceBefore=2, spaceAfter=3)))

                        # 전체 공정 통합 top5 계산
                        _all_scores = {}
                        for _pn in _rp_proc_order:
                            _gp = _g_df[_g_df["process"]==_pn][_avail_s]
                            _op = _o_df[_o_df["process"]==_pn][_avail_s]
                            if _gp.empty or _op.empty: continue
                            _std_p = _rp_sensor[_rp_sensor["process"]==_pn][_avail_s].std().replace(0, 1e-8)
                            _nd = ((_gp.mean() - _op.mean()).abs() / _std_p)
                            for _sc in _avail_s:
                                _key = (_pn, _sc)
                                _all_scores[_key] = float(_nd.get(_sc, 0))

                        _top5_keys = sorted(_all_scores, key=lambda x: _all_scores[x], reverse=True)[:5]
                        _s5_data = [["공정", "센서", "변화", "정상 평균", "불량 평균"]]
                        for (_pn5, _sc5) in _top5_keys:
                            _gp5 = _g_df[_g_df["process"]==_pn5][_sc5].mean()
                            _op5 = _o_df[_o_df["process"]==_pn5][_sc5].mean()
                            if pd.isna(_gp5) or pd.isna(_op5): continue
                            _arrow = "▲ 높음" if _gp5 > _op5 else "▼ 낮음"
                            _s5_data.append([_pn5, _rp_sens_kr.get(_sc5,_sc5), _arrow,
                                             f"{_op5:.1f}", f"{_gp5:.1f}"])

                        _s5t = Table(_s5_data,
                                     colWidths=[_W*0.15, _W*0.2, _W*0.2, _W*0.22, _W*0.23])
                        _s5_style = [
                            ("BACKGROUND",    (0,0), (-1,0),  _rl_colors.HexColor(_dc_h)),
                            ("TEXTCOLOR",     (0,0), (-1,0),  _rl_colors.white),
                            ("FONTNAME",      (0,0), (-1,-1), _rf_name),
                            ("FONTSIZE",      (0,0), (-1,-1), 8),
                            ("ALIGN",         (2,0), (-1,-1), "CENTER"),
                            ("ROWBACKGROUNDS",(0,1), (-1,-1),
                             [_rl_colors.HexColor("#fafafa"), _rl_colors.HexColor("#f0f0f0")]),
                            ("GRID",          (0,0), (-1,-1), 0.3, _rl_colors.HexColor("#cccccc")),
                            ("TOPPADDING",    (0,0), (-1,-1), 3),
                            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                        ]
                        for _ri5 in range(1, len(_s5_data)):
                            _c5 = _rl_colors.HexColor("#c03030") if "▲" in _s5_data[_ri5][2] \
                                  else _rl_colors.HexColor("#2060b0")
                            _s5_style.append(("TEXTCOLOR", (2,_ri5), (2,_ri5), _c5))
                        _s5t.setStyle(TableStyle(_s5_style))
                        _story.append(_s5t)
                        _story.append(Spacer(1, 10))

            # ── 섹션 3 : 불량 제품 목록
            if _rp_vision:
                _rp_defect_rows = [r for r in _rp_vision if r["defect"] not in ("NoDefects","Error")]
                if _rp_defect_rows:
                    _story.append(Paragraph(f"3. 불량 제품 목록 ({len(_rp_defect_rows)}건)", S_H1))
                    _tbl_d = [["제품 ID", "파일명", "불량 유형", "신뢰도"]]
                    for _rr in _rp_defect_rows[:60]:
                        _tbl_d.append([
                            str(_rr.get("product_id","-")),
                            str(_rr.get("filename","-"))[:30],
                            str(_rr.get("defect","-")),
                            f"{_rr.get('confidence',0)*100:.1f}%",
                        ])
                    _dt = Table(_tbl_d, colWidths=[_W*0.23, _W*0.37, _W*0.23, _W*0.17])
                    _dt.setStyle(TableStyle([
                        ("BACKGROUND",    (0,0), (-1,0),  _rl_colors.HexColor("#d4612a")),
                        ("TEXTCOLOR",     (0,0), (-1,0),  _rl_colors.white),
                        ("FONTNAME",      (0,0), (-1,-1), _rf_name),
                        ("FONTSIZE",      (0,0), (-1,0),  8),
                        ("FONTSIZE",      (0,1), (-1,-1), 7.5),
                        ("ROWBACKGROUNDS",(0,1), (-1,-1),
                         [_rl_colors.HexColor("#fafafa"), _rl_colors.HexColor("#f0f0f0")]),
                        ("GRID",          (0,0), (-1,-1), 0.3, _rl_colors.HexColor("#cccccc")),
                        ("ALIGN",         (3,0), (3,-1),  "CENTER"),
                        ("TOPPADDING",    (0,0), (-1,-1), 3),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                    ]))
                    _story.append(_dt)
                    if len(_rp_defect_rows) > 60:
                        _story.append(Paragraph(
                            f"※ 전체 {len(_rp_defect_rows)}건 중 상위 60건만 표시", S_CAP))

            # ── PDF 빌드
            _doc.build(_story)
            _rfb.seek(0)
            st.session_state["t4_pdf_buf"] = _rfb.getvalue()
            st.success("✅ PDF 보고서 생성 완료!")

        except ImportError:
            st.error("❌ reportlab 패키지가 필요합니다. 터미널에서 `pip install reportlab` 을 실행하세요.")
        except Exception as _e4:
            st.error(f"❌ PDF 생성 실패: {_e4}")
            import traceback
            st.code(traceback.format_exc())

    # ── PDF 다운로드 버튼
    if "t4_pdf_buf" in st.session_state:
        _fname4 = f"chocolate_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="⬇ PDF 보고서 다운로드",
            data=st.session_state["t4_pdf_buf"],
            file_name=_fname4,
            mime="application/pdf",
            use_container_width=True,
            key="t4_dl_btn",
        )

