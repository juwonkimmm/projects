import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import plotly.graph_objects as go
import modules
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

modules.create_sidebar()
this_year = st.session_state['year']
current_month = st.session_state['month']

#DataFrame 하이라이트 추가
def highlight_columns(col):
    if col.name in columns_to_color:
        return ['background-color: #f0f0f0'] * len(col)
    return [''] * len(col)

@st.cache_data(ttl=1800)
def create_income_statement_form(year, month):
    index_tuples = [
        ('매출액', ' '),
        ('매출액', '제품등'),
        ('매출액', '부산물'),
        
        ('판매량', ' '),
            
        ('매출원가', ' '),
        ('매출원가', '제품원가'),
        ('매출원가', 'C조건 선임'),
        ('매출원가', '클레임'),
        ('매출원가', '재고평가분'),
        ('매출원가', '단가소급 등'),
    
        ('매출이익', ' '),
        ('매출이익 (%)', ' '),
    
        ('판관비', ' '),
        ('판관비', '인건비'),
        ('판관비', '관리비'),
        ('판관비', '판매비'),
    
        ('영업이익', ' '),
        ('영업이익 (%)', ' '),
    
        ('판매비', ' '),
        ('판매비', '내수운반'),
        ('판매비', '수출개별'),
    
        ('판매량', ' '),
        ('판매량', '내수'),
        ('판매량', '수출')]
        
    hier_index = pd.MultiIndex.from_tuples(index_tuples)

    hier_column = pd.MultiIndex.from_tuples([
        (' ', f'\'{str(this_year-2)[-2:]}년'), (' ', f'\'{str(this_year-1)[-2:]}년'),
        ('전월대비', f'{month-2}월'), ('전월대비', f'{month-1}월'), ('전월대비', ' '),
        ('계획대비', f'{month-2}월계획'), ('계획대비', f'{month-1}월계획'), ('계획대비', ' '),
        (' ', '당월누적')])
    
    #columns = [f'{year-2}년', 
    #            f'{year-1}년', 
    #            f'{month - 2}월',
    #            f'{month - 1}월',
    #            '전월대비', 
    #            '4월계획',
    #            '5월계획',
    #            '계획대비',
    #            '당월누적']
        
    df = pd.DataFrame(0, index=hier_index, columns=hier_column)
    return df

def update_income_form(year, month):
    
    df = create_income_statement_form(year, month)
    
    # secrets.toml에서 직접 데이터 로드
    file_name = st.secrets["sheets"]["f_19"]
    data = pd.read_csv(file_name, thousands=',')
    
    # 데이터 전처리
    data['실적'] = round(data['실적']).astype(float)
    data['월'] = data['월'].astype(str).apply(lambda x: x if '월' in x else x + '월')
    data = data.fillna('')
    
    # 월 데이터를 숫자로 변환하여 처리
    data['월_숫자'] = data['월'].astype(str).str.replace('월', '').astype(int)
    
    # 당월누적 데이터 계산
    temp = data[data['월_숫자'] < month].groupby(['대구분', '소구분'])['실적'].sum()
    if ('판매량', ' ') in temp.index:
        temp[('판매량', ' ')] = temp[('판매량', ' ')]/2
    
    for i in df.index:
        if i in temp.index:
            df.iloc[:, -1] = temp[i]

    # 전월대비, 계획대비 데이터 입력
    for i in df.index:
        month_2_data = data[(data['월_숫자'] == month-2) & (data['대구분'] == i[0]) & (data['소구분'] == i[1])]
        month_1_data = data[(data['월_숫자'] == month-1) & (data['대구분'] == i[0]) & (data['소구분'] == i[1])]
        
        if not month_2_data.empty:
            df.loc[i, ('전월대비', f'{month-2}월')] = month_2_data['실적'].values[0]
            if '계획' in month_2_data.columns:
                df.loc[i, ('계획대비', f'{month-2}월계획')] = month_2_data['계획'].values[0]
        
        if not month_1_data.empty:
            df.loc[i, ('전월대비', f'{month-1}월')] = month_1_data['실적'].values[0]
            if '계획' in month_1_data.columns:
                df.loc[i, ('계획대비', f'{month-1}월계획')] = month_1_data['계획'].values[0]

        if i in temp.index:
            df.loc[i].iloc[:, 8] = temp[i]

    # 계산 로직
    df.iloc[:, 4] = df.iloc[:, 3] - df.iloc[:, 2] #전월대비
    df.iloc[:, 7] = df.iloc[:, 3] - df.iloc[:, 6] #계획대비

    # 퍼센트 계산 및 포맷팅
    df.iloc[11, :] = df.iloc[11, :] * 100
    df.iloc[11, -1] = round((df.iloc[10, -1] / df.iloc[0, -1]) * 100, 1)
    df.iloc[17, :] = df.iloc[17, :] * 100
    df.iloc[17, -1] = round((df.iloc[16, -1] / df.iloc[0, -1]) * 100, 1)

    new_index = list(df.index)
    new_index[3] = ('판매량', '  ')
    df.index = pd.MultiIndex.from_tuples(new_index)

    df.iloc[11, :] = df.iloc[11, :].apply(lambda x: f"{x:.1f}%")
    df.iloc[17, :] = df.iloc[17, :].apply(lambda x: f"{x:.1f}%")
    
    return df

def display_styled_df(df, styles=None, highlight_cols=None):
    """DataFrame에 스타일을 적용하여 화면 중앙에 표시합니다."""

    def highlight_columns(col):
        if col.name in (highlight_cols or []):
            return ['background-color: #f0f0f0'] * len(col)
        return [''] * len(col)

    styled_df = (
        df.style
        .format(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and pd.notnull(x) else x)
        .set_properties(**{'text-align': 'right', 'font-family': 'Noto Sans KR'})
        .apply(highlight_columns, axis=0)
    )
    if styles:
        styled_df = styled_df.set_table_styles(styles)

    table_html = styled_df.to_html(index=True)
    centered_html = f"<div style='display: flex; justify-content: center;'>{table_html}</div>"
    st.markdown(centered_html, unsafe_allow_html=True)

def display_memo(memo_file_key):
    """메모 파일 키를 받아 해당 메모를 화면에 표시합니다."""
    file_name = st.secrets['memos'][memo_file_key]
    try:
        df_memo = pd.read_csv(file_name)
        str_list = df_memo['메모'][0].split('\n')
        html_items = [create_indented_html(s) for s in str_list]
        body_content = "".join(html_items)

        # CSS와 HTML 코드를 함수 내에서 한 번만 정의
        html_code = f"""
        <style>
            .memo-body {{
                font-family: 'Noto Sans KR', sans-serif;
                word-spacing: 5px;
            }}
            .memo-body .indent-0 {{ padding-left: 0px; padding-top: 10px; text-indent: -30px; font-size: 17px; font-weight: bold; }}
            .memo-body .indent-1 {{ padding-left: 20px; padding-top: 5px; text-indent: -10px; font-size: 17px; }}
            .memo-body .indent-2 {{ padding-left: 40px; font-size: 17px; }}
            .memo-body .indent-3 {{ padding-left: 60px; font-size: 12px; }}
            .memo-body p {{ margin: 0.2rem 0; }}
        </style>
        <div class="memo-body">{body_content}</div>
        """
        _, col, _ = st.columns([0.1, 0.8, 0.1]) 
        with col:
            st.markdown(html_code, unsafe_allow_html=True)
    except (FileNotFoundError, KeyError):
        st.warning(f"메모 파일을 찾을 수 없습니다: {memo_file_key}")

def convert_leading_spaces(s):
    stripped_s = s.lstrip(' ')
    num_spaces = len(s) - len(stripped_s)
    return '&nbsp;' * num_spaces + stripped_s

def create_indented_html(s):
    content = s.lstrip(' ')
    num_spaces = len(s) - len(content)
    indent_level = num_spaces // 2
    return f'<p class="indent-{indent_level}">{content}</p>'
    
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
st.markdown('<div class="centered">', unsafe_allow_html=True)

st.markdown(f"## {this_year}년 {current_month}월 손익 분석", unsafe_allow_html=True)
#st.divider()

t1, t2, t3, t4, t5, t6 = st.tabs(['1. 손익요약', '2. 전월 대비 손익차이', '3. 원재료', '4. 제조 가공비', '5. 판매비와 관리비', '6. 성과급 및 격려금'])

# 1. 손익요약 탭
with t1:
    st.markdown("<h4>1. 손익요약</h4>", unsafe_allow_html=True)
    try:
        df_income = update_income_form(this_year, current_month)
        
        # 스타일 정의 (매출 분석과 동일)
        border_rows = [1, 4, 7, 10, 13, 14, 17]
        styles = [{'selector': f'tr:nth-child({row_idx + 2})', 'props': [('border-bottom', '2px solid grey')]} for row_idx
                  in border_rows]
        styles.append({'selector': 'thead tr:last-child th', 'props': [('border-bottom', '2px solid grey')]})
        columns_to_color = [('전월대비', f'{current_month-2}월'), ('전월대비', f'{current_month-1}월'), ('계획대비', f'{current_month-2}월계획'), ('계획대비', f'{current_month-1}월계획')]

        # styled_df 함수 호출 (매출 분석과 동일한 방식)
        display_styled_df(df_income, styles=styles, highlight_cols=columns_to_color)
        
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}")