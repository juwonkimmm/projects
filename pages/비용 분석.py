import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import plotly.graph_objects as go
import modules
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

@st.cache_data(ttl=1800)
def load_data(url):
    data = pd.read_csv(url, thousands=',')
    data['실적'] = round(data['실적']).astype(float)
    data['월'] = data['월'].astype(str).apply(lambda x: x if '월' in x else x + '월')
    data = data.fillna('')
    return data

modules.create_sidebar()
this_year = st.session_state['year']
current_month = st.session_state['month']

#------------------------------------------------------------------------------------------------

custom_css = """
<style>
table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Noto Sans KR', sans-serif;
    font-size: 13px;
    line-height: 1.2;  /* 행 높이 줄임 */
}
th, td {
    padding: 3px 6px;  /* 세로 여백 줄임 */
    text-align: right;
    border: 1px solid #ddd;
    vertical-align: middle;
    word-break: keep-all;
    white-space: pre-wrap;
}
thead {
    background-color: #f2f2f2;
    font-weight: bold;
}
.row_heading { display: none !important; }
.blank { display: none !important; }
</style>
"""

#------------------------------------------------------------------------------------------------

# 화면 구성
st.markdown('<div class="centered">', unsafe_allow_html=True)
st.image("D:\\seah\\pages\\logo.gif", width=200)
st.markdown(f"## {this_year}년 {current_month}월 비용 분석")
t1, t2, t3, t4, t5, t6, t7 = st.tabs(['사용량 원단위 추이_포항', '사용량 원단위 추이_충주1','사용량 원단위 추이_충주2','단가 추이', '월 평균 클레임 지급액','당월 클레임 내역','영업외 비용 내역'])
st.divider()
#------------------------------------------------------------------------------------------------

# =========================
# 사용량 원단위 추이_포항
# =========================
with t1:
    pass

# =========================
#사용량 원단위 추이_충주1
# =========================
# =========================
# 단가 추이
# =========================
with t2:
   
    pass

# =========================
#사용량 원단위 추이_충주2
# =========================
with t3:
    pass

# =========================
#단가 추이
# =========================
with t4:
    pass

# =========================
#월 평균 클레임 지급액
# =========================
with t5:
    st.markdown(f"<h4>1. 클레임 현황</h4>", unsafe_allow_html=True)
    st.markdown(f"<h5>* 월 평균 클레임 지급액</h5>", unsafe_allow_html=True)
    df = modules.update_monthly_claim_form(this_year)

    styled_df = (
            df.style
            .format(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and pd.notnull(x) else x)
            .set_properties(**{'text-align': 'right'})
            .set_properties(**{'font-family': 'Noto Sans KR'})
        )

    table_html = styled_df.to_html(index=True)
    centered_html = f"<div style='display: flex; justify-content: center;'>{table_html}</div>"
    st.markdown(centered_html, unsafe_allow_html=True)

# =========================
#당월 클레임 내역
# =========================
with t6:
    st.markdown(f"<h5>* {current_month}월 클레임 내역</h5>", unsafe_allow_html=True)
    file_name = st.secrets['sheets']['f_48']
    data = load_data(file_name)
    data['실적'] /= 1000000

    df_2 = modules.create_df(this_year, current_month, data, mean = "False", prev_year = 1)

    for i in data['구분2'].unique():
        df_2.loc[(i, ' '), :] = df_2.loc[(i, '불량 보상'), :] + df_2.loc[(i, '선별비'), :]

    df_2.loc[:, '증감'] = df_2.iloc[:, -1] - df_2.iloc[:, -2]

    df_2.loc[('합계', '불량 보상'), :] = df_2.iloc[[0, 3, 6, 9, 12]].sum()
    df_2.loc[('합계', '선별비'), :] = df_2.iloc[[1, 4, 7, 10, 13]].sum()
    df_2.loc[('합계', ' '), :] = df_2.iloc[[2, 5, 8, 11, 14]].sum()

    level1_order = ['선재', '봉강', '부산', '대구', '글로벌', '합계']
    level2_order = [' ', '선별비', '불량 보상']

    df_2.index = pd.MultiIndex.from_arrays([
        pd.Categorical(df_2.index.get_level_values(0), categories=level1_order, ordered=True),
        pd.Categorical(df_2.index.get_level_values(1), categories=level2_order, ordered=True)])

    df_2 = df_2.sort_index()

    styled_df = (
        df_2.style
        .format(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) and pd.notnull(x) else x)
        .set_properties(**{'text-align': 'right'})
        .set_properties(**{'font-family': 'Noto Sans KR'})
    )

    table_html_2 = styled_df.to_html(index=True)
    centered_html_2 = f"<div style='display: flex; justify-content: center;'>{table_html_2}</div>"
    st.markdown(centered_html_2, unsafe_allow_html=True)

# =========================
#영업외 비용 내역
# =========================
with t7:
    pass

# Footer
st.markdown("""
<style>.footer { bottom: 0; left: 0; right: 0; padding: 8px; text-align: center; font-size: 13px; color: #666666;}</style>
<div class="footer">ⓒ 2025 SeAH Special Steel Corp. All rights reserved.</div>
""", unsafe_allow_html=True)