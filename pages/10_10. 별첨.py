import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import plotly.graph_objects as go
import modules

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions (도우미 함수) ---
@st.cache_data(ttl=1800)
def load_data(url):
    """CSV 데이터를 로드하고 기본 전처리를 수행합니다."""
    data = pd.read_csv(url, thousands=',')
    data['실적'] = round(data['실적']).astype(float)
    data['월'] = data['월'].astype(str).apply(lambda x: x if '월' in x else x + '월')
    data = data.fillna('')
    return data

def display_summary_chart(df, key, yaxis1_range, yaxis2_range):
    """실적 요약(막대+꺾은선) 차트를 생성하고 화면에 표시합니다."""
    plot_rows = ['매출액', '판매량', '영업이익']
    df_plot = df.loc[plot_rows].copy()

    # 숫자형 변환
    sales = pd.to_numeric(df_plot.loc['매출액'], errors='coerce')
    profit = pd.to_numeric(df_plot.loc['영업이익'], errors='coerce')

    # 매출이 0이면 영업이익률 0, 아니면 계산
    margin = np.where(sales == 0, 0, (profit / sales) * 100)

    df_plot.loc['영업이익률'] = margin
    df_plot = df_plot.T

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot['매출액'], name='매출액', marker_color='#3b4951',
        width=0.4, text=df_plot['매출액'], texttemplate='%{text:,.0f}',
        textposition='inside', insidetextanchor='middle', insidetextfont=dict(color='white')
    ))
    fig.add_trace(go.Bar(
        x=df_plot.index, y=df_plot['판매량'], name='판매량', marker_color='#e54e2b',
        width=0.4, text=df_plot['판매량'], texttemplate='%{text:,.0f}',
        textposition='inside', insidetextanchor='middle', insidetextfont=dict(color='white')
    ))

    # 텍스트와 hovertemplate를 위한 데이터 준비
    custom_text = [
        f"{profit:,.0f}<br>({margin:.1f}%)"
        for profit, margin in zip(df_plot['영업이익'], df_plot['영업이익률'])
    ]

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['영업이익'], name='영업이익', mode='lines+markers+text',
        text=custom_text, customdata=df_plot['영업이익률'],
        hovertemplate='<b>%{x}</b><br>영업이익: %{y:,.0f}<br>영업이익률: %{customdata:.1f}%<extra></extra>',
        marker=dict(size=8, color='grey'), line=dict(width=3, color='grey'),
        yaxis='y2', textposition="top center", textfont=dict(size=15, color='black')
    ))

    fig.update_layout(
        height=500, font=dict(size=15), bargap=0.2, barmode='group', plot_bgcolor='white',
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=yaxis1_range),
        yaxis2=dict(showticklabels=False, overlaying='y', side='right', showgrid=False, zeroline=False,
                    range=yaxis2_range),
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey', tickfont=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(size=18)),
        margin=dict(t=80, b=20, l=20, r=20)
    )
    _, chart_col, _ = st.columns([0.2, 0.6, 0.2])
    with chart_col:
        st.plotly_chart(fig, use_container_width=True, key=key)


def display_line_chart(df, traces, key):
    """다축 꺾은선 그래프를 생성하고 화면에 표시합니다."""
    df_plot = df.T
    fig = go.Figure()
    layout_options = {}

    for i, trace in enumerate(traces, 1):
        axis_name = 'y' if i == 1 else f'y{i}'
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[trace['name']], name=trace['name'][1],
            mode='lines+markers+text', marker=dict(size=8, color=trace['color']),
            line=dict(width=3, color=trace['color']), yaxis=axis_name,
            text=df_plot[trace['name']], textposition=trace.get('textposition', 'top center'),
            textfont=dict(size=15, color='black'), texttemplate='%{text:,.1f}',
            hovertemplate=f"{trace['name'][1]}: %{{y}}<extra></extra>"
        ))
        axis_config = dict(showticklabels=False, showgrid=False, zeroline=False, range=trace['range'])
        if i > 1:
            axis_config.update(overlaying='y', side='right')
        layout_options[f'yaxis{"" if i == 1 else i}'] = axis_config

    fig.update_layout(
        height=500, font=dict(size=15), plot_bgcolor='white',
        xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey', tickfont=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(size=18)),
        margin=dict(t=80, b=20, l=20, r=20),
        **layout_options
    )
    _, chart_col, _ = st.columns([0.2, 0.6, 0.2])
    with chart_col:
        st.plotly_chart(fig, use_container_width=True, key=key)
# --- Main Streamlit App ---
modules.create_sidebar()
this_year = st.session_state['year']
current_month = st.session_state['month']

st.markdown(f"## {this_year}년 {current_month}월 별첨")

t1, t2, t3, t4, t5 = st.tabs(['실적요약', '가격차이', '환율 추이', '손익계산서', '유형별 손익분석 (수정정상원가 기반)'])
all_dfs = modules.update_performance_form(this_year, current_month)

# 전체 실적요약
with t1:
    st.markdown("<h4>1) 전체 실적요약 (해외법인 포함)</h4>", unsafe_allow_html=True)
    df_list = list(all_dfs.values())
    total_df = df_list[0].copy()

    for df in df_list[1:]:
        total_df += df

    plot_rows = ['매출액', '판매량', '영업이익']
    plot_columns = total_df.columns
    df_plot = total_df.loc[plot_rows, plot_columns].copy()

    # 숫자형 변환
    sales = pd.to_numeric(df_plot.loc['매출액', :], errors='coerce')
    profit = pd.to_numeric(df_plot.loc['영업이익', :], errors='coerce')

    # 매출액이 0이면 영업이익률 0으로 처리
    margin = np.where(sales == 0, 0, (profit / sales) * 100)

    df_plot.loc['영업이익률', :] = margin

    # % 표시용 문자열 변환
    df_plot.loc['영업이익률', :] = (
        pd.to_numeric(df_plot.loc['영업이익률', :], errors='coerce')
        .round(1)
        .astype(str)
        + "%"
    )

    df_plot = df_plot.T


    display_summary_chart(total_df, key="total_summary", yaxis1_range=[0, 150000], yaxis2_range=[-10000, 8000])
    st.divider()

# 본사 실적요약

    st.markdown("<h4>2) 본사 실적요약</h4>", unsafe_allow_html=True)
    display_summary_chart(all_dfs['본사'], key="hq_summary", yaxis1_range=[0, 150000], yaxis2_range=[-10000, 8000])
    st.divider()

# 중국법인 실적요약

    st.markdown("<h4>3) 중국법인 실적요약</h4>", unsafe_allow_html=True)
    display_summary_chart(all_dfs['중국'], key="cn_summary", yaxis1_range=[0, 40000], yaxis2_range=[-1000, 1500])
    st.divider()

# 태국법인 실적요약

    st.markdown("<h4>4) 태국법인 실적요약</h4>", unsafe_allow_html=True)
    display_summary_chart(all_dfs['태국'], key="th_summary", yaxis1_range=[0, 10000], yaxis2_range=[-300, 400])
    st.divider()

# 환율 추이
with t2:
    st.markdown("<h4>1) 포스코 대 JFE 가격 차이</h4>", unsafe_allow_html=True)
    df_raw = modules.create_df(this_year, current_month, load_data(st.secrets['sheets']['f_93']), mean="False",
                               prev_year=1, prev_month=6)
    df_plot = df_raw.loc[('가격차이', ['탄소강', '합금강']), df_raw.columns]

    traces = [
        {'name': ('가격차이', '탄소강'), 'color': '#3b4951', 'range': [-100, 400]},
        {'name': ('가격차이', '합금강'), 'color': '#e54e2b', 'range': [-100, 400], 'textposition': 'bottom center'}
    ]
    display_line_chart(df_plot, traces, key="price_diff_chart")
    st.divider()

with t3:
    st.markdown("<h4>1) 환율 추이(연/월 평균 환율기준)</h4>", unsafe_allow_html=True)
    df = modules.create_df(this_year, current_month, load_data(st.secrets['sheets']['f_94']), mean="False", prev_year=1)
    df_plot = df.loc[('환율추이', ['USD', 'CNH', 'THB']), df.columns]

    traces = [
        {'name': ('환율추이', 'USD'), 'color': '#3b4951', 'range': [1300, 1500]},
        {'name': ('환율추이', 'CNH'), 'color': '#e54e2b', 'range': [160, 230], 'textposition': 'bottom center'},
        {'name': ('환율추이', 'THB'), 'color': '#0070c0', 'range': [30, 100]}
    ]
    display_line_chart(df_plot, traces, key="exchange_rate_chart")
    st.divider()
# 가격 차이


# Footer
st.markdown("""
<style>.footer { bottom: 0; left: 0; right: 0; padding: 8px; text-align: center; font-size: 13px; color: #666666;}</style>
<div class="footer">ⓒ 2025 SeAH Special Steel Corp. All rights reserved.</div>
""", unsafe_allow_html=True)