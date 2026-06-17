# 🔍 초콜릿 비전 품질 검사 AI 시스템

YOLO / EfficientNet 비전 모델로 초콜릿 불량을 자동 검사하고, RAG 기반 AI 챗봇으로 원인·조치를 질의응답하는 **Streamlit 웹 애플리케이션**입니다.

---

## 📋 주요 기능

| 탭 | 설명 |
|---|---|
| 🤖 **비전 모델 생성** | 불량 이미지를 업로드해 YOLO(Detection) 또는 EfficientNet/MobileNetV3/ResNet(Classification) 모델 학습 |
| 🔍 **비전 모델로 불량 검사** | 학습된 모델로 제품 이미지를 일괄 검사하고 결과 TXT 자동 생성 |
| 💬 **AI 챗봇 질의응답** | 검사 결과 · 매뉴얼 · 센서 데이터를 BGE-M3로 임베딩 후 Ollama LLM으로 RAG 질의응답 |
| 📄 **비전 품질 분석 보고서** | 검사 결과 · 챗봇 분석을 PDF 보고서로 자동 생성 |

---

## 🍫 불량 유형

| 클래스 | 설명 |
|---|---|
| `Bloom` | 블루밍 — 표면 흰색·회색 얼룩 (균열 없음) |
| `Crack` | 균열 — 표면·내부 균열 (블루밍 없음) |
| `Crack_Bloom` | 복합 불량 — 균열 + 블루밍 동시 발생 |
| `NoDefects` | 정상 |

> `Bloom`, `Crack`, `Crack_Bloom`은 완전히 독립된 클래스입니다. 혼동 없이 분리하여 학습하세요.

---

## 🤖 지원 비전 모델

### Classification (이미지 분류)

| 모델 | 특징 |
|---|---|
| EfficientNet-B0 | 경량·고성능, 권장 기본 모델 |
| MobileNetV3-Small | 초경량, 엣지 디바이스 적합 |
| ResNet-18 | 안정적인 베이스라인 |

### Detection (객체 탐지, YOLO)

| 모델 | 특징 |
|---|---|
| YOLOv5nu | 경량 빠른 추론 |
| YOLOv8n | 정확도·속도 균형 |
| YOLO11n | 최신 아키텍처 |

---

## 💬 AI 챗봇 구조 (RAG)

```
검사 결과 TXT + 매뉴얼 TXT + 센서 CSV
            ↓
      텍스트 청킹 (chunk_size 조절 가능)
            ↓
   BGE-M3 임베딩 (BAAI/bge-m3)
   FlagEmbedding 또는 sentence-transformers 사용
            ↓
     코사인 유사도 기반 Top-K 검색
            ↓
   Ollama LLM (qwen3:8b 기본, 변경 가능)
            ↓
         한국어 답변 스트리밍
```

**추천 질문 예시**

- "전체 불량률 요약해줘"
- "Crack 불량 제품 목록 알려줘"
- "블루밍 예방 조건은?"
- "온도·습도 경고 기준 알려줘"
- "정기 점검 항목 알려줘"

---

## 📂 디렉토리 구조

```
chocolate-vision/
├── chocolate_vision.py    # 메인 앱
├── manual.txt             # 품질 관리 매뉴얼 (챗봇 자동 로드)
├── sensor_data.csv        # 센서 데이터 (챗봇 자동 로드, 선택)
└── requirements.txt
```

> `manual.txt`와 `sensor_data.csv`를 앱과 같은 폴더에 두면 챗봇 탭에서 자동으로 불러옵니다.

---

## 🛠️ 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
streamlit
pandas
numpy
matplotlib
torch
torchvision
ultralytics
Pillow
FlagEmbedding
sentence-transformers
reportlab
joblib
```

> 패키지가 일부 없어도 앱은 실행됩니다. 해당 기능 진입 시 설치 안내가 표시됩니다.

### 2. Ollama 설치 및 LLM 모델 다운로드 (챗봇 사용 시)

```bash
# Ollama 설치: https://ollama.com
ollama serve
ollama pull qwen3:8b   # 기본 모델 (앱에서 변경 가능)
```

### 3. 앱 실행

```bash
streamlit run chocolate_vision.py
```

---

## 📸 이미지 촬영 가이드

정확한 모델 학습을 위해 아래 조건을 지켜 이미지를 수집하세요.

- ✅ 실제 검수 작업대 위에서 촬영 (현장과 동일한 배경·조명)
- 📐 제품을 분류대 중앙에 놓고 위에서 수직으로, 거리 20~40 cm 유지
- 💡 해상도 640 px 이상, 그림자 최소화, **클래스당 30장 이상** 수집
- ⚠️ 사무실·창고 등 현장과 다른 장소에서 촬영하면 모델 정확도가 크게 낮아집니다

---

## 📊 입력 데이터 형식

### 학습 이미지

- 형식: `.jpg` / `.jpeg` / `.png`
- 구성: 정상 이미지 + 불량 이미지 클래스별 업로드
- YOLO Detection 사용 시: YOLO 형식 라벨(`.txt`) + 이미지 ZIP

### 센서 CSV (챗봇 컨텍스트용)

| 컬럼 | 필수 | 설명 |
|---|---|---|
| `product_id` | ✅ | 제품 고유 ID |
| `defect_label` | ✅ | 불량 유형 |
| `process` | ✅ | 공정명 |
| `timestamp` | ✅ | 측정 시각 |
| `temperature_c` 외 센서 컬럼 | — | 각 공정 센서값 |

### 매뉴얼 TXT

- `manual.txt`를 앱과 같은 폴더에 두면 자동 로드
- 없으면 챗봇 탭에서 직접 업로드 가능

---

## 📄 PDF 보고서 구성

1. **불량 현황 요약** — 총 검사 수 · 불량률 표 및 차트
2. **불량 유형별 상세** — 유형별 검사 이미지 샘플 · 탐지 박스
3. **센서 이상 분석** — 공정별 센서 경고·위험 항목
4. **챗봇 분석 요약** — 마지막 AI 답변 전문 포함

---

## 🖥️ 환경 요건

| 항목 | 권장 |
|---|---|
| Python | 3.9 이상 |
| OS | Windows / macOS / Linux |
| GPU | CUDA 지원 GPU (선택, CPU 동작 가능) |
| RAM | 8 GB 이상 (BGE-M3 로드 시 4 GB 추가) |
| 한글 폰트 | 맑은 고딕 (Windows) / NanumGothic (Linux) / AppleSDGothicNeo (macOS) |

---

