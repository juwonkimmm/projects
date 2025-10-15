import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import modules  # modules.create_board_summary_table, modules.create_defect_summary_pohang 사용

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# =========================
# 공통 테이블 렌더 (인덱스 숨김 + 중복 컬럼 안전)
# =========================
def display_styled_df(df, styles=None, highlight_cols=None):
    """
    - 행 멀티인덱스는 reset_index()로 컬럼 승격 → 왼쪽 숫자 인덱스 제거
    - reset_index()로 생길 수 있는 '중복 컬럼명' 자동 고유화
    - Styler.hide(axis="index")로 인덱스 헤더까지 숨김
    """
    # 1) 멀티인덱스(행) → 컬럼 승격
    df_for_style = df.reset_index()

    # 2) 중복 컬럼명 자동 고유화 (예: '', '', '구분'...)
    new_cols = []
    seen = {}
    for c in df_for_style.columns:
        c_str = str(c)
        if c_str in seen:
            seen[c_str] += 1
            new_cols.append(f"{c_str}.{seen[c_str]}")  # ex) ''.1, ''.2
        else:
            seen[c_str] = 0
            new_cols.append(c_str)
    df_for_style.columns = new_cols

    # 3) 강조 컬럼 스타일
    hi_set = set(map(str, (highlight_cols or [])))
    def highlight_columns(col):
        return ['background-color: #f0f0f0'] * len(col) if str(col.name) in hi_set else [''] * len(col)

    # 4) 스타일 지정 + 인덱스 완전 숨김
    styled_df = (
        df_for_style.style
        .format(lambda x: f"{x:,.0f}" if isinstance(x, (int, float, np.integer, np.floating)) and pd.notnull(x) else x)
        .set_properties(**{'text-align': 'right', 'font-family': 'Noto Sans KR'})
        .apply(highlight_columns, axis=0)
        .hide(axis="index")
    )
    if styles:
        styled_df = styled_df.set_table_styles(styles)

    # 5) 렌더
    table_html = styled_df.to_html()
    centered_html = f"<div style='display: flex; justify-content: center;'>{table_html}</div>"
    st.markdown(centered_html, unsafe_allow_html=True)

# =========================
# 날짜 선택 사이드바
# =========================
this_year = datetime.today().year
current_month = datetime.today().month

def _date_update_callback():
    st.session_state.year = st.session_state.year_selector
    st.session_state.month = st.session_state.month_selector

def create_sidebar():
    with st.sidebar:
        st.title("날짜 선택")
        if 'year' not in st.session_state:
            st.session_state.year = this_year
        if 'month' not in st.session_state:
            st.session_state.month = current_month

        st.selectbox('년(Year)', range(2020, 2031),
                     key='year_selector',
                     index=st.session_state.year - 2020,
                     on_change=_date_update_callback)

        st.selectbox('월(Month)', range(1, 13),
                     key='month_selector',
                     index=st.session_state.month - 1,
                     on_change=_date_update_callback)

        st.info(f"선택된 날짜: {st.session_state.year}년 {st.session_state.month}월")

create_sidebar()

# =========================
# 안전 로더 (원본 '톤' 단위 그대로)
# =========================
@st.cache_data(ttl=1800)
def load_f40(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, dtype=str)

    # 실적 → float
    if '실적' in df.columns:
        s = df['실적'].str.replace(',', '', regex=False)
        df['실적'] = pd.to_numeric(s, errors='coerce').fillna(0.0)
    else:
        df['실적'] = 0.0

    # 월 → Int64
    if '월' in df.columns:
        m = (df['월'].astype(str).str.replace('월','',regex=False)
                           .str.replace('.','',regex=False).str.strip()
                           .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NULL': np.nan}))
        df['월'] = pd.to_numeric(m, errors='coerce').astype('Int64')
    else:
        df['월'] = pd.Series([pd.NA]*len(df), dtype='Int64')

    # 연도 → Int64 (2자리면 20xx)
    if '연도' in df.columns:
        y = (df['연도'].astype(str).str.extract(r'(\d{4}|\d{2})')[0]
                           .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NULL': np.nan}))
        y = y.apply(lambda v: f"20{v}" if isinstance(v, str) and len(v)==2 else v)
        df['연도'] = pd.to_numeric(y, errors='coerce').astype('Int64')
    else:
        df['연도'] = pd.Series([pd.NA]*len(df), dtype='Int64')

    # 구분 → 문자열
    for c in ['구분1', '구분2', '구분3', '구분4']:
        if c in df.columns: df[c] = df[c].fillna('').astype(str)
        else: df[c] = ''

    return df

@st.cache_data(ttl=1800)
def load_defect(url: str) -> pd.DataFrame:
    """부적합 데이터 로더"""
    df = pd.read_csv(url, dtype=str)
    # 숫자 형변환
    for c in ['연도', '월', '실적']:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    for c in ['구분1','구분2','구분3','구분4']:
        if c in df.columns: df[c] = df[c].fillna('').astype(str)
        else: df[c] = ''
    return df

# =========================
# UI 본문
# =========================
year = int(st.session_state['year'])
month = int(st.session_state['month'])

st.image("logo.gif", width=200)
st.markdown(f"## {year}년 {month}월 생산 분석")
t1, t2, t3 = st.tabs(['전체 생산실적', '부적합 발생내역_포항공장','부적합 발생내역_충주 1,2공장'])
st.divider()

# =========================
# 전체 생산실적
# =========================
with t1:
    st.markdown("<h4>1) 전체 생산실적</h4>", unsafe_allow_html=True)

    # 표 우측 상단 단위
    unit = "<div style='text-align:right; font-size:14px; color:#666;'>[단위: 톤]</div>"
    st.markdown(unit, unsafe_allow_html=True)

    try:
        raw40 = load_f40(st.secrets['sheets']['f_40'])

        df_board = modules.create_board_summary_table(
            year, month, raw40,
            base_year=year,           # 현재 선택연도 기준
            prev_year_for_avg=year-1  # 전년 평균
        )

        # 숫자 포맷팅
        df_show = df_board.copy()

        def _fmt_diff(x):
            try:
                xi = int(round(float(x)))
                return f"({abs(xi):,})" if xi < 0 else f"{xi:,}"
            except Exception:
                return x

        def _fmt_pct(x):
            try:
                return f"{float(x):.1f}%"
            except Exception:
                return x

        for c in df_show.columns:
            if c == "전월대비":
                df_show[c] = df_show[c].apply(_fmt_diff)
            elif c == "%":
                df_show[c] = df_show[c].apply(_fmt_pct)
            else:
                df_show[c] = df_show[c].apply(
                    lambda v: f"{int(v):,}" if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v) else v
                )

        # 등급별 보더 스타일 유사 적용
        border_rows = []
        for i, (g, sub) in enumerate(df_board.index, start=1):
            if g in ["CHQ", "CD", "STS", "BTB", "PB", "합계"]:
                border_rows.append(i)

        styles_prod = []
        styles_prod.extend([
            {'selector': f'tbody tr:nth-child({r})', 'props': [('border-bottom', '3px solid grey !important')]}
            for r in border_rows
        ])
        styles_prod.append({'selector': 'thead tr:last-child th', 'props': [('border-bottom', '3px solid grey')]})
        styles_prod.append({'selector': 'th.row_heading', 'props': [('border-right', '2px solid #ddd'),
                                                                    ('background-color', '#fff')]})

        highlight_cols = [f"'{str(year-1)[-2:]}년 월평균", f"'{str(year)[-2:]}년 월평균", "전월대비", "%"]

        display_styled_df(df_show, styles=styles_prod, highlight_cols=highlight_cols)

        # 표 좌측 하단 집계기준
        foot = "<div style='text-align:left; font-size:13px; color:#666;'>※ 집계기준 : 원재 투입량 + 비가공 + 제품 재가공</div>"
        st.markdown(foot, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"사업부/공장 요약 표를 표시하는 중 오류가 발생했습니다: {e}")

# =========================
# 부적합 발생내역 - 포항
# =========================



with t2:
    st.markdown("<h4>2) 부적합 발생내역 (포항)</h4>", unsafe_allow_html=True)
    try:
        df_src = load_defect(st.secrets['sheets']['f_41_42'])

        # 1월~선택월까지 전부 컬럼화
        df_pohang = modules.create_defect_summary_pohang(
            year, month, df_src, months_window=tuple(range(1, month+1)), plant_name="포항"
        )

        # 인덱스 머리글: 구분
        if isinstance(df_pohang.index, pd.MultiIndex):
            n = df_pohang.index.nlevels
            if n == 3:
                df_pohang.index = df_pohang.index.set_names(['구분','\u2007', '\u2009' ])
            elif n == 2:
                df_pohang.index = df_pohang.index.set_names(['\u2007', '구분'])
            else:
                df_pohang.index = df_pohang.index.set_names(['구분'])

        # 굵은 보더(섹션 경계)
        thick_rows_zero_based = [2, 5, 8]
        styles_def = []
        styles_def.extend([
            {'selector': f'tbody tr:nth-child({r+1})',
             'props': [('border-bottom', '3px solid grey !important')]}
            for r in thick_rows_zero_based
        ])
        styles_def.append({'selector': 'thead tr:last-child th',
                           'props': [('border-bottom', '3px solid grey')]})

        hl_cols = [f"{str(year-1)[-2:]}년 월평균", f"{str(year)[-2:]}년 목표", '합계', '월평균']

        display_styled_df(df_pohang, styles=styles_def, highlight_cols=hl_cols)

    except Exception as e:
        st.error(f"포항 부적합 표 생성 중 오류가 발생했습니다: {e}")


# =========================
# 부적합 발생내역 - 충주 1,2공장 (자리만)
# =========================
with t3:
    st.markdown("<h4>3) 부적합 발생내역 (충주 1,2공장)</h4>", unsafe_allow_html=True)
    try:
        df_src = load_defect(st.secrets['sheets']['f_41_42'])

        # 1월~선택월 전체를 컬럼으로
        all_months = tuple(range(1, month + 1))
        df_cjj = modules.create_defect_summary_chungju(
            year, month, df_src, months_window=all_months,
            plant1_name="충주", plant2_name="충주2"
        )

                # 인덱스 머리글: 구분
        if isinstance(df_cjj.index, pd.MultiIndex):
            n = df_cjj.index.nlevels
            if n == 3:
                df_cjj.index = df_cjj.index.set_names(['구분','\u2007', '\u2009' ])
            elif n == 2:
                df_cjj.index = df_cjj.index.set_names(['\u2007', '구분'])
            else:
                df_cjj.index = df_cjj.index.set_names(['구분'])

        thick_rows_zero_based = [2, 5]
        styles_def = []
        styles_def.extend([
            {'selector': f'tbody tr:nth-child({r+1})',
             'props': [('border-bottom', '3px solid grey !important')]}
            for r in thick_rows_zero_based
        ])
        styles_def.append({'selector': 'thead tr:last-child th',
                           'props': [('border-bottom', '3px solid grey')]})

        # 강조 컬럼
        hl_cols = [f"{str(year-1)[-2:]}년 월평균", f"{str(year)[-2:]}년 목표", '합계', '월평균']

        display_styled_df(df_cjj, styles=styles_def, highlight_cols=hl_cols)

    except Exception as e:
        st.error(f"충주 1,2공장 부적합 표 생성 중 오류가 발생했습니다: {e}")


# =========================
# Footer
# =========================
st.markdown("""
<style>.footer { bottom: 0; left: 0; right: 0; padding: 8px; text-align: center; font-size: 13px; color: #666666;}</style>
<div class="footer">ⓒ 2025 SeAH Special Steel Corp. All rights reserved.</div>
""", unsafe_allow_html=True)
