# 📊 VGA Price Vision: AI 기반 그래픽카드 시세 예측 솔루션

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

<br>

> **"과거의 데이터를 통해 미래의 가치를 읽다."**
> <br>
> 딥러닝(LSTM)을 활용하여 그래픽카드(VGA)의 과거 시계열 데이터를 분석하고, 
> 향후 30일간의 가격 변동 범위를 예측하는 전문 AI 대시보드입니다.

</div>

<br>

## 📺 프로젝트 데모 (Project Demo)

<div align="center">
  [여기에 영상을 드래그 앤 드롭 하세요]

</div>

<br>

---

<details>
<summary><b>🔍 1. 핵심 기능 (Key Features) - [클릭]</b></summary>

<br>

### 🛠 데이터 파이프라인
- **Crawling:** `Crawlers/` 내 스크립트를 통해 주기적인 하드웨어 시세 데이터 수집.
- **Preprocessing:** 이동 평균(Smoothing) 기법을 통한 노이즈 캔슬링 및 이상치 제거.

### 🤖 딥러닝 예측 모델
- **Architecture:** 2-Stack **LSTM** 구조를 통한 시계열 패턴 학습.
- **Accuracy:** RTX 4060 기준 **R² Score 0.9846** 달성.
- **Confidence Interval:** 단순 선형 예측이 아닌, RMSE 기반 **예측 신뢰 구간** 시각화.

### 📈 데이터 인사이트 (EDA)
- 요일별/월별 시세 빈도 분석 및 변수 간 상관관계(Heatmap) 제공.
- 가격 변동 범위의 이상치를 탐지하는 박스플롯 분석 기능.

</details>

<br>

<details>
<summary><b>🛠 2. 기술 스택 (Tech Stack) - [클릭]</b></summary>

<br>

| 분류 | 기술 스택 |
| :--- | :--- |
| **Language** | Python 3.12 |
| **AI/ML** | TensorFlow, Keras, Scikit-learn |
| **Data** | Pandas, NumPy, Joblib |
| **Visualization** | Streamlit, Matplotlib, Seaborn |
| **DevOps** | Git, VS Code |

</details>

<br>

<details>
<summary><b>📂 3. 프로젝트 구조 (Project Structure) - [클릭]</b></summary>

<br>

```bash
vga-price-forecaster/
├── Crawlers/             # 데이터 수집 스크립트
├── Dataset/              # 원본 및 정제 데이터셋
│   └── VGA_Total/
├── models/               # 학습 완료된 AI 모델(.h5) 및 스케일러(.pkl)
├── report/               # 분석 리포트 및 시각화 결과물
├── src/                  # 메인 소스 코드
│   ├── app_test.py       # Streamlit 대시보드 실행 파일
│   └── vga_trainer.ipynb # 모델 학습 및 검증 노트북
├── assets/               # README용 이미지/GIF 저장소
├── requirements.txt      # 의존성 패키지 리스트
└── README.md             # 프로젝트 문서
