# AI 기반 VGA(그래픽카드) 시세 분석 및 예측 시스템

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

## 1. 프로젝트 개요
본 프로젝트는 국내 주요 하드웨어 커뮤니티의 과거 시세 데이터를 수집 및 전처리하고, 시계열 데이터 학습에 특화된 **LSTM(Long Short-Term Memory)** 딥러닝 모델을 활용하여 특정 그래픽카드(VGA)의 향후 30일간 가격 변동 추이를 예측하는 분석 시스템입니다.

단순한 선형 예측을 넘어, 데이터의 변동성(Volatility)을 고려한 신뢰 구간(Confidence Interval)을 시각화함으로써 사용자의 합리적인 구매 의사결정을 지원하는 것을 목적으로 합니다.

---

## 2. 시스템 시연 (System Demo)

<div align="center">
  <video src="assets/main_demo.mp4" width="100%" controls autoplay muted loop></video>
  <br>
  <i>(영상 재생이 원활하지 않을 경우 <a href="assets/main_demo.mp4">원본 파일</a>을 확인하십시오.)</i>
</div>

---

<details open>
<summary><b>3. 주요 기능 및 기술적 특징 (Key Features)</b></summary>
<br>

### 3.1 데이터 파이프라인 (Data Pipeline)
* **데이터 수집:** `Crawlers` 모듈을 통한 일자별/모델별 시세 데이터 확보.
* **전처리(Preprocessing):**
    * 결측치(Missing Value) 보간 및 이상치(Outlier) 필터링.
    * 이동 평균(Moving Average) 기법을 적용하여 일시적 가격 급등락(Noise) 제거 및 추세선 추출.

### 3.2 예측 모델링 (AI Modeling)
* **모델 구조:** 2-Stack LSTM 레이어 구조를 채택하여 장/단기 시계열 패턴 학습.
* **학습 파라미터:** 과거 30일(Window Size) 데이터를 입력으로 하여 익일 가격을 예측하는 Many-to-One 방식 적용.
* **성능 지표:** RTX 4060 모델 기준 **R² Score 0.9846**, **RMSE(평균 제곱근 오차) 약 5,700원**의 예측 정밀도 확보.

### 3.3 분석 시각화 (Visualization)
* **신뢰 구간 시각화:** 모델의 RMSE를 기반으로 예측 가격의 상한/하한 범위(Band)를 도출하여 불확실성 표현.
* **EDA(탐색적 데이터 분석):** 요일별/월별 가격 빈도 분석 및 변수 간 상관관계(Correlation Heatmap) 제공.

</details>

<br>

<details>
<summary><b>4. 시스템 구조 및 개발 환경 (System Architecture)</b></summary>
<br>

### 4.1 디렉토리 구조
```bash
vga-price-forecaster/
├── Crawlers/             # 웹 크롤링 및 데이터 수집 스크립트
├── Dataset/              # 원본 데이터(Raw) 및 전처리 데이터(CSV)
│   └── VGA_Total/
├── models/               # 학습된 LSTM 모델(.h5) 및 Scaler 객체(.pkl)
├── report/               # 프로젝트 결과 보고서 및 분석 자료
├── src/                  # 애플리케이션 소스 코드
│   ├── app_test.py       # Streamlit 대시보드 엔트리 포인트
│   └── vga_trainer.ipynb # 모델 학습, 검증 및 평가 스크립트
├── assets/               # README 리소스 (영상, 이미지)
├── requirements.txt      # Python 의존성 패키지 목록
└── README.md             # 프로젝트 기술 문서


