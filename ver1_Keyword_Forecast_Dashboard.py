import pandas as pd
from prophet import Prophet
import streamlit as st
import plotly.graph_objs as go
import matplotlib as mpl

# 한글 폰트 설정
mpl.rc('font', family='Malgun Gothic')

# 데이터 불러오기 및 전처리
file_path = '월별 검색량 데이터.csv'
df_raw = pd.read_csv(file_path)
df_raw = df_raw.rename(columns={'월': 'ds'})
df_raw['ds'] = pd.to_datetime(df_raw['ds'])
df_melted = df_raw.melt(id_vars=['ds'], var_name='keyword', value_name='search_volume')

# 최근 3개월 대비 이전 3개월 성장률 계산
recent_data = df_raw.tail(3)
previous_data = df_raw.iloc[-6:-3]
growth_rates = {}
for keyword in df_raw.columns[1:]:
    recent_avg = recent_data[keyword].mean()
    previous_avg = previous_data[keyword].mean()
    if previous_avg > 0:
        growth = (recent_avg - previous_avg) / previous_avg * 100
    else:
        growth = 0
    growth_rates[keyword] = growth

top_keywords = pd.Series(growth_rates).sort_values(ascending=False).head(5).index.tolist()

# Prophet 예측 함수
def get_forecast(keyword):
    df_target = df_melted[df_melted['keyword'] == keyword][['ds', 'search_volume']]
    df_target = df_target.rename(columns={'search_volume': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.5)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_target)
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    return df_target, forecast

# Streamlit UI 구성
st.title("🔍 키워드 검색량 예측 대시보드")

selected_keyword = st.selectbox("키워드를 선택하세요:", top_keywords)
months = st.slider("표시할 기간 (개월):", min_value=6, max_value=24, step=3, value=12)

# 예측 수행
df_target, forecast = get_forecast(selected_keyword)
cutoff = forecast['ds'].max() - pd.DateOffset(months=months)
forecast_filtered = forecast[forecast['ds'] >= cutoff]
df_target_filtered = df_target[df_target['ds'] >= cutoff]

# 예측 그래프
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df_target_filtered['ds'], y=df_target_filtered['y'],
                                  mode='lines+markers', name='실제값'))
fig_forecast.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['yhat'],
                                  mode='lines', name='예측값'))
fig_forecast.update_layout(title=f'{selected_keyword} 검색량 예측')
st.plotly_chart(fig_forecast)

# 트렌드 그래프
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['trend'],
                               mode='lines', name='트렌드'))
fig_trend.update_layout(title=f'{selected_keyword} 트렌드 분석')
st.plotly_chart(fig_trend)

# 상위 키워드 비교 그래프
st.subheader("📈 상위 키워드 검색량 비교")
fig_compare = go.Figure()
for keyword in top_keywords:
    df = df_melted[df_melted['keyword'] == keyword][['ds', 'search_volume']]
    df = df[df['ds'] >= cutoff]
    fig_compare.add_trace(go.Scatter(x=df['ds'], y=df['search_volume'],
                                     mode='lines', name=keyword))
fig_compare.update_layout(title='상위 키워드 검색량 비교')
st.plotly_chart(fig_compare)
