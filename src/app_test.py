import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. 설정 및 라이브러리 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="VGA 시세 예측 시스템", layout="wide")

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# [경로 설정] src 폴더 기준, 한 단계 상위(../) 폴더를 루트로 설정
# -----------------------------------------------------------------------------
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
PROJECT_ROOT = os.path.dirname(current_dir)

# 데이터 및 모델 경로 연결
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'Dataset')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# TensorFlow 로드 확인
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    st.error("⚠️ TensorFlow 모듈이 설치되지 않았습니다.")

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 처리 함수
# -----------------------------------------------------------------------------
@st.cache_data
def get_vga_list():
    """VGA_Total 폴더에서 그래픽카드 모델명 리스트를 반환"""
    target_folder = os.path.join(BASE_DATA_DIR, "VGA_Total")
    
    if not os.path.exists(target_folder):
        return [], target_folder
    
    files = sorted([f for f in os.listdir(target_folder) if f.endswith('.csv')])
    if not files:
        return [], target_folder

    try:
        latest = files[-1]
        path = os.path.join(target_folder, latest)
        try: df = pd.read_csv(path, encoding='utf-8')
        except: df = pd.read_csv(path, encoding='cp949')
            
        def cleaner(name):
            if not isinstance(name, str): return None
            # VGA 정규식 적용
            match = re.search(r'(RTX|RX|GTX)\s?\d{3,4}\s?(Ti|SUPER|XT|XTX|GRE)?', name, re.I)
            return match.group(0).strip() if match else None

        if 'Name' in df.columns:
            return sorted(df['Name'].apply(cleaner).dropna().unique().tolist()), target_folder
        return [], target_folder
    except:
        return [], target_folder

@st.cache_data
def load_data(folder_path, target_model):
    """선택한 VGA 모델의 과거 시세 데이터를 통합 로드 및 전처리"""
    all_data = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for f in files:
        path = os.path.join(folder_path, f)
        df_tmp = None
        for enc in ['utf-8', 'cp949']:
            try: df_tmp = pd.read_csv(path, encoding=enc); break
            except: continue
            
        if df_tmp is None or 'Name' not in df_tmp.columns: continue
        
        rows = df_tmp[df_tmp['Name'].str.contains(target_model, na=False, case=False)]
        cols = [c for c in df_tmp.columns if re.match(r'\d{4}-\d{2}-\d{2}', c)]
        
        for col in cols:
            p = pd.to_numeric(rows[col].astype(str).str.replace(',', '').str.extract('(\d+)')[0], errors='coerce')
            valid = p[p > 10000] # VGA 가격 필터링 (1만원 이상)
            if not valid.empty:
                all_data.append({'Date': col.split(' ')[0], 'Price': valid.mean()})
    
    if not all_data: return None

    df = pd.DataFrame(all_data).groupby('Date')['Price'].mean().reset_index()
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date_dt')
    
    df['Year'] = df['Date_dt'].dt.year
    df['Month'] = df['Date_dt'].dt.month
    df['DayOfWeek'] = df['Date_dt'].dt.dayofweek
    df['Price_Raw'] = df['Price']
    df['Price_Smooth'] = df['Price'].rolling(window=3, min_periods=1).mean()
    
    return df

# -----------------------------------------------------------------------------
# 3. 메인 로직 (VGA 전용)
# -----------------------------------------------------------------------------
def main():
    if not HAS_TF:
        st.error("TensorFlow가 설치되지 않아 예측 기능을 사용할 수 없습니다.")
        return

    # [수정] 사이드바 제거 및 VGA 고정 로직
    # 1. 타이틀 출력
    st.title("📉 VGA(그래픽카드) 시세 예측 시스템")
    st.markdown("데이터 기반의 **인공지능(LSTM)** 모델이 향후 시세를 분석하고 예측합니다.")
    st.markdown("---")
    
    # 2. 모델 목록 가져오기 (VGA 고정)
    model_list, folder_path = get_vga_list()

    if model_list:
        # 기본 선택값 설정 (RTX 4060 우선)
        idx = 0
        default_target = "RTX 4060"
        for i, name in enumerate(model_list):
            if default_target in name:
                idx = i
                break
        
        # [수정] 메인 화면에 Selectbox 배치
        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            selected_model = st.selectbox("👇 분석할 그래픽카드를 선택하세요", model_list, index=idx)
        with col_sel2:
            st.empty() # 여백
            
    else:
        st.error(f"❌ 'Dataset/VGA_Total' 폴더에 데이터가 없거나 경로가 잘못되었습니다.")
        st.stop()

    # 데이터 로드
    with st.spinner(f'📊 {selected_model} 시세 데이터를 분석 중입니다...'):
        df_final = load_data(folder_path, selected_model)

    # 3. AI 모델 및 스케일러 경로 탐색 (VGA 고정)
    safe_name = selected_model.replace(" ", "_")
    category = "vga" # 소문자 고정
    
    path_specific = os.path.join(MODEL_DIR, f"{category}_{safe_name}_model.h5")
    path_generic = os.path.join(MODEL_DIR, f"{category}_model.h5")
    final_model_path = path_specific if os.path.exists(path_specific) else (path_generic if os.path.exists(path_generic) else None)
    
    scaler_candidates = [
        os.path.join(MODEL_DIR, f"{category}_{safe_name}_scaler.pkl"),
        os.path.join(MODEL_DIR, f"{category}_scaler.pkl"),
        os.path.join(MODEL_DIR, f"{category}_model.pkl")
    ]
    final_scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
    
    has_model = (final_model_path is not None) and (final_scaler_path is not None)

    # 4. 분석 결과 시각화
    if df_final is not None:
        st.divider()
        st.header(f"📌 {selected_model} 분석 리포트")
        
        # [섹션 1] 모델 성능 및 정확도
        SEQ_LENGTH = 30
        scaled_data = None
        model_ai = None
        scaler_ai = None

        if has_model:
            try:
                model_ai = load_model(final_model_path)
                scaler_ai = joblib.load(final_scaler_path)
                scaled_data = scaler_ai.transform(df_final[['Price_Smooth']])
                
                if len(scaled_data) > SEQ_LENGTH:
                    X_val = np.array([scaled_data[i:i+SEQ_LENGTH] for i in range(len(scaled_data)-SEQ_LENGTH)])
                    y_pred = scaler_ai.inverse_transform(model_ai.predict(X_val, verbose=0))
                    y_actual = df_final['Price_Smooth'].values[SEQ_LENGTH:]
                    
                    st.subheader("1. AI 모델 신뢰도")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R² Score (정확도)", f"{r2_score(y_actual, y_pred):.4f}")
                    m2.metric("MAE (평균오차)", f"{mean_absolute_error(y_actual, y_pred):,.0f}원")
                    m3.metric("MSE", f"{mean_squared_error(y_actual, y_pred):,.0f}")
                    m4.metric("RMSE", f"{np.sqrt(mean_squared_error(y_actual, y_pred)):,.0f}원")
                else:
                    st.warning("⚠️ 데이터가 부족하여 성능 평가를 수행할 수 없습니다.")
            except Exception as e:
                st.error(f"모델 로드 중 오류 발생: {e}")
                has_model = False
        else:
            st.info("💡 학습된 모델 파일(.h5)을 찾을 수 없습니다. (먼저 학습을 진행해주세요)")

        st.markdown("---")
        
        # [섹션 2] 통계 그래프
        st.subheader("2. 주요 데이터 분포")
        
        # 그래프 크기(figsize)를 모두 (10, 6)으로 통일
        COMMON_FIG_SIZE = (10, 6)
        
        # 1행: 가격 분포 / 요일별 빈도 (2단 분할)
        r1_c1, r1_c2 = st.columns(2)
        
        with r1_c1:
            fig, ax = plt.subplots(figsize=COMMON_FIG_SIZE)
            sns.histplot(df_final['Price_Raw'], kde=True, ax=ax, color='skyblue')
            ax.set_title("가격대 분포 (Histogram)")
            st.pyplot(fig)
            
        with r1_c2:
            fig, ax = plt.subplots(figsize=COMMON_FIG_SIZE)
            sns.countplot(data=df_final, x='DayOfWeek', hue='DayOfWeek', palette='viridis', legend=False, ax=ax)
            ax.set_title("요일별 데이터 수")
            st.pyplot(fig)

        # 2행: 월별 빈도 / 상관관계 (2단 분할)
        r2_c1, r2_c2 = st.columns(2)
        
        with r2_c1:
            fig, ax = plt.subplots(figsize=COMMON_FIG_SIZE)
            sns.countplot(data=df_final, x='Month', hue='Month', palette='magma', legend=False, ax=ax)
            ax.set_title("월별 데이터 수")
            st.pyplot(fig)
            
        with r2_c2:
            # 상관관계 분석용 임시 변수 생성
            df_corr = df_final.copy()
            df_corr['DaysFromStart'] = (df_corr['Date_dt'] - df_corr['Date_dt'].min()).dt.days
            target_cols = ['Price', 'Year', 'Month', 'DayOfWeek', 'DaysFromStart']
            valid_cols = [c for c in target_cols if c in df_corr.columns]
            
            fig, ax = plt.subplots(figsize=COMMON_FIG_SIZE)
            sns.heatmap(df_corr[valid_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title("변수 간 상관관계 (Heatmap)")
            st.pyplot(fig)

        # 3행: 이상치 분석 (2단 분할 중 왼쪽 사용)
        r3_c1, r3_c2 = st.columns(2)
        
        with r3_c1:
            fig, ax = plt.subplots(figsize=COMMON_FIG_SIZE)
            sns.boxplot(y=df_final['Price_Raw'], color='lightcoral', ax=ax)
            ax.set_title("가격 이상치 분석 (Boxplot)")
            st.pyplot(fig)
            
        with r3_c2:
            # 빈 공간을 두어 레이아웃 균형 유지
            st.empty()

        st.markdown("---")

        # [섹션 3] 시세 추이 및 미래 예측
        st.subheader("3. 시세 추이 및 미래 예측")
        tab1, tab2 = st.tabs(["📉 과거 시세 데이터", "🔮 미래 시세 예측 (30일)"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_final['Date_dt'], df_final['Price_Raw'], label='Raw Price', alpha=0.5)
            ax.plot(df_final['Date_dt'], df_final['Price_Smooth'], label='Trend (Smooth)', color='red', linewidth=2)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
        with tab2:
            if has_model and scaled_data is not None:
                # ---------------------------------------------------------
                # [업그레이드] 예측 범위(Confidence Interval) 시각화 로직
                # ---------------------------------------------------------
                PREDICT_DAYS = 30  # 예측 기간
                
                # 1. 미래 예측 수행
                last_seq = scaled_data[-SEQ_LENGTH:]
                future_preds = []
                
                for _ in range(PREDICT_DAYS):
                    nxt = model_ai.predict(last_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
                    future_preds.append(nxt[0])
                    last_seq = np.append(last_seq[1:], nxt, axis=0)
                
                # 2. 스케일 복원 및 날짜 생성
                future_prices = scaler_ai.inverse_transform(future_preds)
                last_date = df_final['Date_dt'].max()
                future_dates = [last_date + timedelta(days=i) for i in range(1, PREDICT_DAYS + 1)]
                
                # 3. [핵심] 예측 범위 계산 (RMSE 활용)
                # 모델이 가진 평균 오차(RMSE)만큼 위아래로 여유를 둡니다.
                # (약 5,700원 정도의 오차 범위를 반영)
                rmse_val = np.sqrt(mean_squared_error(y_actual, y_pred)) 
                
                # numpy 배열로 변환하여 연산
                pred_mean = future_prices.flatten()
                upper_bound = pred_mean + rmse_val  # 최대 예상가
                lower_bound = pred_mean - rmse_val  # 최소 예상가

                # 4. 그래프 그리기 (범위 색칠 추가)
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # (1) 과거 데이터
                ax.plot(df_final['Date_dt'][-60:], df_final['Price_Smooth'].values[-60:], label='Past 60 Days', color='#4A90E2', linewidth=2)
                
                # (2) 미래 예측 선 (중앙값)
                ax.plot(future_dates, pred_mean, color='#FF4B4B', label='Predicted Trend', linewidth=2, linestyle='--')
                
                # (3) [NEW] 예측 범위 색칠하기 (Fill Between)
                ax.fill_between(future_dates, lower_bound, upper_bound, color='#FF4B4B', alpha=0.15, label=f'Confidence Range (±{int(rmse_val):,}KRW)')
                
                # 스타일링
                ax.set_title(f"향후 {PREDICT_DAYS}일 시세 예측 범위", fontsize=16, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(loc='upper left')
                
                # X축 날짜 포맷팅 예쁘게
                plt.xticks(rotation=0)
                
                st.pyplot(fig)
                
                # 5. 텍스트 리포트
                end_price = pred_mean[-1]
                diff = end_price - future_prices[0][0]
                
                st.info(f"""
                **💡 분석 결과:**
                향후 **{PREDICT_DAYS}일 뒤** 예상 가격은 약 **{int(end_price):,}원** 입니다.
                데이터의 변동성을 고려했을 때, 최저 **{int(lower_bound[-1]):,}원**에서 최고 **{int(upper_bound[-1]):,}원** 사이에서 움직일 것으로 전망됩니다.
                """)
                
            else:
                st.write("모델이 없어 미래를 예측할 수 없습니다.")
    else:
        st.error("데이터를 로드하는 데 실패했습니다.")

if __name__ == "__main__":
    main()