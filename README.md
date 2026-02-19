# VGA Price Vision
### AI-Powered Graphics Card Price Prediction Solution

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## 1. Project Overview

**VGA Price Vision**
- 변동성이 큰 그래픽카드(VGA) 시장의 시세를 추적하고, 딥러닝 기반의 시계열 예측을 통해 미래 가격 흐름을 제공하는 솔루션 
- 그래픽 카드 별 가격 추이를 및 30일 내의 미래 예측 가격을 안내하여 구매 타이밍에 도움을 주는 프로그램

<br>

<div align="center">
  <img src="assets/20251218_VGA 시세 예측기.gif" width="90%" alt="Project Demo" />
</div>

<br>

## 2. Key Features

### Data Pipeline & Preprocessing
* **Automated Crawling:** `Crawlers/` 모듈을 통해 주요 하드웨어 커뮤니티 및 마켓의 시세 데이터를 수집

### Deep Learning Model (LSTM)
* **Architecture:** 시계열 데이터의 장기 의존성(Long-term dependency) 학습에 최적화된 **2-Stack LSTM** 구조를 설계
* **Performance:** RTX 4060 모델 기준 **R² Score 0.9846**의 높은 예측 정확도를 달성
* **Confidence Interval:** 단순 점 추정이 아닌 RMSE 기반의 **95% 신뢰 구간**을 함께 시각화하여 예측의 불확실성을 보정

### Exploratory Data Analysis (EDA)
* **Market Insight:** 요일별/월별 시세 빈도 분석을 통해 특정 시기의 가격 변동 패턴을 도출
* **Correlation Analysis:** 히트맵(Heatmap)과 박스플롯(Boxplot)을 통해 변수 간의 상관관계 및 가격 변동 범위를 시각적으로 제공

<br>

## 3. Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **AI / ML** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Visualization** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=flat-square&logo=Matplotlib&logoColor=black) |
| **DevOps** | ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) |

<br>

## 4. Project Structure

```bash
vga-price-forecaster/
├── Crawlers/               # 데이터 수집 스크립트 (Web Scrapers)
├── Dataset/                # 원본 및 전처리 완료 데이터셋
│   └── VGA_Total/
├── models/                 # 학습된 모델(.h5) 및 스케일러(.pkl) 아티팩트
├── report/                 # 데이터 분석 리포트 및 시각화 결과
├── src/                    # 메인 애플리케이션 소스 코드
│   ├── app_test.py         # Streamlit 대시보드 엔트리 포인트
│   └── vga_trainer.ipynb   # 모델 학습 및 검증용 Jupyter Notebook
├── assets/                 # README 리소스
├── requirements.txt        # 프로젝트 의존성 패키지 목록
└── README.md               # 프로젝트 문서
