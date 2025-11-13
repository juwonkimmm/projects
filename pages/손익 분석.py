import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import modules  
import io
import re
from itertools import groupby  
warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# =========================
# 공통 테이블 렌더 (인덱스 숨김 + 중복 컬럼 안전)
# =========================


import re, io, pandas as pd
from urllib.request import urlopen, Request






def rowspan_like_for_index(blocks, level=2, header_rows=1):
    """
    멀티인덱스(행) 열에서, 연속된 행들을 '한 칸처럼' 보이게 하는 CSS 스타일을 만들어줍니다.
    - blocks: [(start_data_row, end_data_row), ...]  # 데이터 기준 0-based, 양끝 포함
    - level:  대상 인덱스 레벨 번호 (구분 레벨이 보통 2)
    - header_rows: tbody 위에 끼운 가짜 헤더 수(보통 1)
    반환: set_table_styles에 append할 dict 리스트
    """
    styles = []
    to_nth = lambda r: r + header_rows + 1  # 0-based 데이터행 → tbody nth-child(1-based)

    for start, end in blocks:
        top = to_nth(start)
        mid = [to_nth(r) for r in range(start + 1, end)]
        bot = to_nth(end)

        # 시작행: 아래 경계 제거
        styles.append({
            'selector': f'tbody tr:nth-child({top}) th.row_heading.level{level}',
            'props': [('border-bottom', '0')]
        })
        # 중간행들: 위/아래 경계 제거 + 텍스트 숨김
        for r in mid:
            styles.append({
                'selector': f'tbody tr:nth-child({r}) th.row_heading.level{level}',
                'props': [('border-top', '0'), ('border-bottom', '0'),
                          ('color', 'transparent'), ('text-shadow', 'none')]
            })
        # 끝행: 위 경계 제거
        styles.append({
            'selector': f'tbody tr:nth-child({bot}) th.row_heading.level{level}',
            'props': [('border-top', '0')]
        })
    return styles

def with_inline_header_row(df: pd.DataFrame,
                           index_names=('', '', '구분'),
                           index_values=('', '', '구분')) -> pd.DataFrame:
    """
    멀티인덱스(행) 위에 '같은 행 높이'로 컬럼명을 보여주기 위해
    본문 첫 행에 '헤더용 가짜 행'을 삽입한다.
    - index_names: df.index.names 를 덮어쓸 이름 (마지막만 '구분'으로 보이게)
    - index_values: 가짜 행의 인덱스 값 튜플 (마지막 칸에 '구분' 텍스트 배치)
    """
    # 1) 원본 인덱스 이름 정리
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.set_names(index_names)
    else:
        df.index.name = index_names[-1]

    # 2) 헤더용 1행(컬럼명 그대로 출력) 만들기
    hdr = pd.DataFrame([list(df.columns)], columns=df.columns)
    if isinstance(df.index, pd.MultiIndex):
        hdr.index = pd.MultiIndex.from_tuples([index_values], names=index_names)
    else:
        hdr.index = pd.Index([index_values[-1]], name=index_names[-1])

    # 3) 본문 위에 합치기 (hdr가 첫 행이 됨)
    df2 = pd.concat([hdr, df], axis=0)
    return df2

def display_styled_df(
    df,
    styles=None,
    highlight_cols=None,
    already_flat=False,
    applymap_rules=None,   # 👈 추가: (func, (row_labels, col_labels)) 리스트
):
    """
    - already_flat=True: df가 이미 index 없는 평평한 형태(= reset_index 완료)라고 가정
    - applymap_rules: [(func, (row_indexer, col_indexer)), ...]
        * row_indexer, col_indexer는 '라벨 기반' 인덱서(= df.index/df.columns에서 뽑은 값)여야 함
        * 예) [(neg_red_func, (df.index[2:], df.columns[4:]))]
    """
    import numpy as np
    import pandas as pd

    if already_flat:
        df_for_style = df.copy()
    else:
        df_for_style = df.reset_index()

    # (중복 컬럼명 고유화)
    new_cols, seen = [], {}
    for c in df_for_style.columns:
        c_str = str(c)
        seen[c_str] = seen.get(c_str, 0) + 1
        new_cols.append(c_str if seen[c_str] == 1 else f"{c_str}.{seen[c_str]-1}")
    df_for_style.columns = new_cols

    hi_set = set(map(str, (highlight_cols or [])))
    def highlight_columns(col):
        return ['background-color: #f0f0f0'] * len(col) if str(col.name) in hi_set else [''] * len(col)

    styled_df = (
        df_for_style.style
        # 숫자는 천단위, 문자열은 그대로(괄호포맷 등은 상위에서 문자열로 만들어졌다고 가정)
        .format(lambda x: f"{x:,.0f}" if isinstance(x, (int,float,np.integer,np.floating)) and pd.notnull(x) else x)
        .set_properties(**{'text-align':'right','font-family':'Noto Sans KR'})
        .apply(highlight_columns, axis=0)
        .hide(axis="index")  # 👈 인덱스 완전 숨김
    )

    if styles:
        styled_df = styled_df.set_table_styles(styles)

    if applymap_rules:
        for func, subset in applymap_rules:
            rows, cols = subset  # 라벨 기반 인덱서여야 함
            styled_df = styled_df.applymap(func, subset=pd.IndexSlice[rows, cols])

    st.markdown(
        f"<div style='display:flex;justify-content:center'>{styled_df.to_html()}</div>",
        unsafe_allow_html=True
    )



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
        m = (df['월'].astype(str).str.replace('월', '', regex=False)
             .str.replace('.', '', regex=False).str.strip()
             .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NULL': np.nan}))
        df['월'] = pd.to_numeric(m, errors='coerce').astype('Int64')
    else:
        df['월'] = pd.Series([pd.NA] * len(df), dtype='Int64')

    # 연도 → Int64 (2자리면 20xx)
    if '연도' in df.columns:
        y = (df['연도'].astype(str).str.extract(r'(\d{4}|\d{2})')[0]
             .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NULL': np.nan}))
        y = y.apply(lambda v: f"20{v}" if isinstance(v, str) and len(v) == 2 else v)
        df['연도'] = pd.to_numeric(y, errors='coerce').astype('Int64')
    else:
        df['연도'] = pd.Series([pd.NA] * len(df), dtype='Int64')

    # 구분 → 문자열
    for c in ['구분1', '구분2', '구분3', '구분4']:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)
        else:
            df[c] = ''
    return df

@st.cache_data(ttl=1800)
def load_defect(url: str) -> pd.DataFrame:
    """부적합 데이터 로더"""
    df = pd.read_csv(url, dtype=str)
    # 숫자 형변환
    for c in ['연도', '월', '실적']:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    for c in ['구분1', '구분2', '구분3', '구분4']:
        if c in df.columns:
            df[c] = df[c].fillna('').astype(str)
        else:
            df[c] = ''
    return df

# =========================
# UI 본문
# =========================
year = int(st.session_state['year'])
month = int(st.session_state['month'])

st.markdown(f"## {year}년 {month}월 손익 분석")

t1, t2, t3, t4, t5, t6, t7 = st.tabs(['1. 손익요약', '2. 전월 대비 손익차이', '3. 원재료', '4. 제조 가공비', '5. 판매비와 관리비', '6. 성과급 및 격려금', '7. 통상임금'])


with t1:

    st.markdown("<h4>1) 손익요약 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)

    try:
        file_name = st.secrets["sheets"]["f_19"]
        df_src = pd.read_csv(file_name, dtype=str)   # [구분1,구분2,구분3,구분4,연도,월,실적]

        # ====== 계산 : 선택 월 기준으로 동적 생성 ======
        body = modules.create_profit_month_block_table(
            df_raw=df_src,
            year=int(st.session_state['year']),
            month=int(st.session_state['month'])
        )

        # ====== 표시 포맷 ======
        def fmt_amt(x):
            if pd.isna(x): return ""
            try:
                v = float(x)
            except Exception:
                return str(x)
            return f"({abs(int(round(v))):,})" if v < 0 else f"{int(round(v)):,}"

        def fmt_pct(x):
            if pd.isna(x): return ""
            try:
                v = float(x)
            except Exception:
                return str(x)
            return f"{v:,.1f}"

        disp = body.copy()   
        assert "구분" in disp.columns, "'구분' 컬럼이 필요합니다."

        # 수치 컬럼 자동 추출
        num_cols = [c for c in disp.columns if c != "구분"]

        # 퍼센트 행 마스크: '구분' 값이 "(%)"로 끝나는 행
        pct_mask = disp["구분"].astype(str).str.endswith("(%)")

        # 숫자형 변환 + 포맷
        for c in num_cols:
            disp[c] = pd.to_numeric(disp[c], errors="coerce")
            disp.loc[~pct_mask, c] = disp.loc[~pct_mask, c].apply(fmt_amt)
            disp.loc[ pct_mask, c] = disp.loc[ pct_mask, c].apply(fmt_pct)


        SPACER = "__spacer__"
        disp.insert(0, SPACER, "")


        cols = disp.columns.tolist()
        c_idx = {c:i for i,c in enumerate(cols)}


        body_cols = [c for c in body.columns if c != "구분"]


        col_map = {
            "prev_year": next((c for c in body_cols if c.startswith("'") and "년" in c), None),  # 첫 번째 'yy년
        }

        def _find(label_contains):
            return next((c for c in body_cols if label_contains in c), None)

        col_23 = next((c for c in body_cols if c.startswith("'") and "년" in c), None)                 # 전전년
        col_24 = next((c for c in body_cols if c != col_23 and c.startswith("'") and "년" in c), None) # 전년
        col_pm = next((c for c in body_cols if c.endswith("월") and "계획" not in c), None)            # 전월
        col_m  = next((c for c in body_cols if c != col_pm and c.endswith("월") and "계획" not in c), None) # 당월
        col_diff     = _find("전월대비")
        col_pm_plan  = next((c for c in body_cols if c.endswith("월계획") or c.endswith("월계획(②)") and c != col_m), None)
        col_m_plan   = next((c for c in body_cols if c.endswith("월계획(②)")), None) or _find("월계획(②)")
        col_gap      = _find("계획대비")
        col_acc      = _find("당월누적")

        # 상단 리본 라벨
        yy = str(int(st.session_state['year']))[-2:]
        mm = int(st.session_state['month'])
        top_label = f"'{yy} {mm}월"

        # 1행에 둘 컬럼들 / 2행에 둘 컬럼들
        row1_cols = [col_23, col_24, col_diff, col_gap, col_acc]                  
        row2_cols = [col_pm, col_m, col_pm_plan, col_m_plan]                       

        # ── 가짜 헤더 2행 구성 ──
        hdr1 = [''] * len(cols)   # 상단 그룹 라벨( '23년, '24년, 전월대비, 계획대비, 당월누적 )
        hdr2 = [''] * len(cols)   # 하단 실 라벨( 전월, 당월, 전월계획, 당월계획(②) ) + '구분', SPACER

        hdr1[c_idx[SPACER]] = '구분'
        hdr1[c_idx['구분']] = ''

        # 1행 라벨(원하는 컬럼만)
        for k in [col_23, col_24, col_diff, col_gap, col_acc]:
            if k in c_idx: hdr1[c_idx[k]] = k

        # 2행 라벨
        for k in [col_pm, col_m, col_pm_plan, col_m_plan]:
            if k in c_idx: hdr2[c_idx[k]] = k

        # 나머지는 공백
        for k in body_cols:
            if k in c_idx and k not in [col_23, col_24, col_diff, col_gap, col_acc, col_pm, col_m, col_pm_plan, col_m_plan]:
                hdr1[c_idx[k]] = ''
                hdr2[c_idx[k]] = ''

        hdr_df = pd.DataFrame([hdr1, hdr2], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)


        # ── 스타일 ──
        styles = [
            {'selector': 'thead', 'props': [('display','none')]},

            # 헤더 1·2 행
            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'), ('padding','8px 8px'),  ('line-height','1'),   ('font-weight','600')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'), ('padding','10px 8px'), ('line-height','1.4'), ('font-weight','600')]},

            # 스페이서 열 (1열)
            {'selector': 'tbody td:nth-child(1)', 'props': [('min-width','100px'), ('width','100px'), ('white-space','nowrap'), ('border-right','0')]},

            # '구분' 열 좌측 정렬 (2열)
            {'selector': 'tbody tr td:nth-child(2)', 'props': [('text-align','left'), ('white-space','nowrap')]},
            {'selector': 'tbody td:nth-child(1)','props': [('text-align','left'), ('white-space','nowrap')]}

        ]





        disp_vis.iloc[2, 1] = "" ; disp_vis.iloc[2, 0]  = "매출액"
        disp_vis.iloc[5, 1] = "" ; disp_vis.iloc[5, 0]  = "판매량"
        disp_vis.iloc[6, 1] = "" ; disp_vis.iloc[6, 0]  = "매출원가"
        disp_vis.iloc[12,1] = "" ; disp_vis.iloc[12,0] = "매출이익"
        disp_vis.iloc[13,1] = "" ; disp_vis.iloc[13,0] = "(%)"
        disp_vis.iloc[14,1] = "" ; disp_vis.iloc[14,0] = "판관비"
        disp_vis.iloc[18,1] = "" ; disp_vis.iloc[18,0] = "영업이익"
        disp_vis.iloc[19,1] = "" ; disp_vis.iloc[19,0] = "(%)"
        disp_vis.iloc[20,1] = "" ; disp_vis.iloc[20,0] = "판매비"
        disp_vis.iloc[23,1] = "" ; disp_vis.iloc[23,0] = "판매량"


        spacer_rules1 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(3)',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in (1,2)
        ]

        styles += spacer_rules1

        spacer_rules2 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(4)',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in (1,2)
        ]

        styles += spacer_rules2

        spacer_rules3 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(7)',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in (1,2)
        ]

        styles += spacer_rules3

        spacer_rules4 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(10)',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in (1,2)
        ]

        styles += spacer_rules4

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                'props': [('border-top','3px solid gray !important')]
               
            }
            for r in range (3,12)
        ]

        styles += spacer_rules5

        spacer_rules6 = [
            {
                'selector': f'tbody tr:nth-child(3) td:nth-child({r})',
                'props': [('border-top','3px solid gray !important')]
               
            }
            for r in range (1,12)
        ]

        styles += spacer_rules6

        spacer_rules7 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({r})',
                'props': [('border-top','3px solid gray !important')]
               
            }
            for r in (5,6,8,9)
        ]

        styles += spacer_rules7

        spacer_rules8 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({r})',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in (6,9)
        ]

        styles += spacer_rules8

        spacer_rules9 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({r})',
                'props': [('border-right','2px solid white !important')]
               
            }
            for r in (5,8)
        ]

        styles += spacer_rules9

        spacer_rules10 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({r})',
                'props': [('border-top','2px solid white !important')]
               
            }
            for r in (3,4,7,10,11)
        ]

        styles += spacer_rules10

        spacer_rules11 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-bottom','3px solid gray !important')]
               
            }
            for r in (5,6,12,14,18,20,23)
        ]

        styles += spacer_rules11
        
        spacer_rules12_1 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                'props': [('border-bottom','2px solid white !important')]
               
            }
            for r in range (1,26)
        ]

        styles += spacer_rules12_1

        
        spacer_rules12 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                'props': [('border-bottom','3px solid gray !important')]
               
            }
            for r in (5,6,12,14,18,20,23)
        ]

        styles += spacer_rules12

        spacer_rules13 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-bottom','3px solid gray !important')]
               
            }
            for r in (3,7,15,21,24)
        ]

        styles += spacer_rules13

        
        spacer_rules14 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-left','2px solid white !important')]
               
            }
            for r in range (1,27)
        ]

        styles += spacer_rules14

        spacer_rules15 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-left','3px solid gray !important')]
               
            }
            for r in (4,5,8,9,10,11,12,16,17,18,22,23,25,26)
        ]

        styles += spacer_rules15

        
        spacer_rules16 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-bottom','2px solid white !important')]
               
            }
            for r in (1,4,8,9,10,11,13,16,17,19,22,25)
        ]

        styles += spacer_rules16

        spacer_rules17 = [
            {
                'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                'props': [('border-right','2px solid white !important')]
               
            }
            for r in (5,6,8,9)
        ]

        styles += spacer_rules17

        spacer_rules18 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in range(3,27)
        ]

        styles += spacer_rules18



        # ── 렌더 ──
        display_styled_df(
            disp_vis,
            styles=styles,
            already_flat=True
        )


    except Exception as e:
        st.error(f"손익요약 생성 중 오류: {e}")


# with t2:

#     st.markdown("<h4>1) 전월대비 손익차이 </h4>", unsafe_allow_html=True)
#     st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)

#     st.divider()

#     st.markdown("<h4>2) 수출 환율 차이 </h4>", unsafe_allow_html=True)
#     st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)

#     st.divider()


import re, io, pandas as pd
from urllib.request import urlopen, Request

# 로더(경로/URL 모두) + 천단위 콤마 제거




def resolve_period(df: pd.DataFrame, sel_y: int, sel_m: int):
    d = df.copy()
    d["연도"] = pd.to_numeric(d["연도"], errors="coerce").astype("Int64")
    d["월"]   = pd.to_numeric(d["월"],   errors="coerce").astype("Int64")
    d = d.dropna(subset=["연도","월"])
    periods = set(zip(d["연도"].astype(int), d["월"].astype(int)))
    if (sel_y, sel_m) in periods:
        return sel_y, sel_m, False
    ly = int(d["연도"].max())
    lm = int(d[d["연도"]==ly]["월"].max())
    return ly, lm, True


with t2:


    st.markdown("<h4>1) 전월대비 손익차이 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<h4>2) 수출 환율 차이 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)
    try:
        # 1) 데이터 로드
        file_name = st.secrets["sheets"]["f_21"]
        df_src = pd.read_csv(file_name)  # [구분1,구분2,연도,월,실적]

        # 2) 선택 연월(그대로 사용)
        use_y = int(st.session_state["year"])
        use_m = int(st.session_state["month"])

        # 3) 연산 (전월 vs 선택월)
        body, prev_lab, curr_lab, usd_delta, usd_effect = modules.fx_export_table(
            df_long=df_src, year=use_y, month=use_m
        )

        # === 숫자 포맷 ===
        disp = body.copy()
        for c in disp.columns:
            if c == "구분": continue
            disp[c] = pd.to_numeric(disp[c], errors="coerce")

        def fmt_rate(x): return "" if pd.isna(x) else f"{x:,.2f}"
        def fmt_diff(x): return "" if pd.isna(x) else f"{x:,.1f}"
        def fmt_int(x):  return "" if pd.isna(x) else f"{int(round(x)):,}"

        rate_cols = [c for c in disp.columns if c.endswith("환율")]
        diff_cols = ["차이단가"]
        int_cols  = [c for c in disp.columns if c not in (["구분"] + rate_cols + diff_cols)]
        for c in rate_cols: disp[c] = disp[c].apply(fmt_rate)
        for c in diff_cols: disp[c] = disp[c].apply(fmt_diff)
        for c in int_cols:  disp[c] = disp[c].apply(fmt_int)

        # === 열 순서 강제 ===
        block_prev = [f"{prev_lab}_중량", f"{prev_lab}_외화공급가액", f"{prev_lab}_환율", f"{prev_lab}_원화공급가액"]
        block_curr = [f"{curr_lab}_중량", f"{curr_lab}_외화공급가액", f"{curr_lab}_환율", f"{curr_lab}_원화공급가액"]
        tail_cols  = ["차이단가","영향금액"]
        ordered = ["구분"] + [c for c in block_prev if c in disp.columns] + [c for c in block_curr if c in disp.columns] + tail_cols
        disp = disp[ordered]

        # === 가짜행 2개 생성 ===
        SPACER = "__spacer__"
        disp.insert(0, SPACER, "")  # 좌측 여백(“구분” 머리 넣을 자리)

        # 1행(상단 그룹라벨) — 월을 "중량"과 "차이단가" 위에만 표시
        hdr1 = []
        for c in disp.columns:
            if c == SPACER:
            #     hdr1.append("구분")
            # elif c == "구분":
                hdr1.append("")
            elif (c in block_prev) and c.endswith("_중량"):
                hdr1.append(prev_lab)
            elif (c in block_curr) and c.endswith("_중량"):
                hdr1.append(curr_lab)
            elif c == "차이단가":
                hdr1.append(curr_lab)   
            else:
                hdr1.append("")

        lbl_fx  = "외화공급\n가액"     
        lbl_krw = "원화공급\n가액"
        hdr2 = []
        for c in disp.columns:
            if c in (SPACER): hdr2.append("구분")
            elif c.endswith("_중량"): hdr2.append("중량")
            elif c.endswith("_외화공급가액"): hdr2.append(lbl_fx)
            elif c.endswith("_환율"): hdr2.append("환율")
            elif c.endswith("_원화공급가액"): hdr2.append(lbl_krw)
            elif c in tail_cols: hdr2.append("차이단가" if c == "차이단가" else "영향금액")
            else: hdr2.append("")

        hdr_df = pd.DataFrame([hdr1, hdr2], columns=disp.columns)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)

        styles = [
            {'selector': 'thead', 'props': [('display','none')]},

            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'),('font-weight','700'),('padding','6px 8px')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'),('font-weight','700'),('padding','8px 8px')]},
            # 본문 우측정렬
            {'selector': 'tbody tr:nth-child(n+3) td', 'props': [('text-align','right')]},
            # 좌측 두 칸(스페이서, 구분) 정렬
            {'selector': 'tbody tr td:nth-child(1)', 'props': [('text-align','left'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(2)', 'props': [('text-align','left'),('white-space','nowrap')]},
        ]


        for r in (6, 10):
            styles += [
                {'selector': f'tbody tr:nth-child(1) td:nth-child({r})', 'props':[('border-right','3px solid gray !important')]},
                {'selector': f'tbody tr:nth-child(2) td:nth-child({r})', 'props':[('border-right','3px solid gray !important')]},
                {'selector': f'tbody tr:nth-child(n+3) td:nth-child({r})', 'props':[('border-right','3px solid gray !important')]}
            ]



        
        spacer_rules1 = [
            {
                'selector':'tbody tr:nth-child(2) td', 
                'props':[('border-bottom','3px solid gray !important')]
               
            }
            for r in (1,2)
        ]

        styles += spacer_rules1


        spacer_rules2 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-left','2px solid white !important')]
               
            }
            for r in range(1,7)
        ]

        styles += spacer_rules2

        
        
        spacer_rules3 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in range(1,7)
        ]

        styles += spacer_rules3

        
        spacer_rules3 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in range(1,7)
        ]

        styles += spacer_rules3

        spacer_rules4 = [
            {
                'selector':'tbody tr:nth-child(1) td', 
                'props':[('border-right','2px solid white !important')]
               
            }
            for r in (3,4,5,7,8,9,11)
        ]

        styles += spacer_rules4

        spacer_rules5 = [
            {
                'selector':'tbody tr:nth-child(1) td', 
                'props':[('border-top','3px solid gray !important')]
               
            }
            for r in range(1,13)
        ]

        styles += spacer_rules5


        spacer_rules6 = [
            {
                'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                'props': [('border-bottom','2px solid white !important')]
               
            }
            for r in (1,2)
        ]

        styles += spacer_rules6


        display_styled_df(disp_vis, styles=styles, already_flat=True)

        # 주석문장
        effect = usd_effect/10000.0
        updown = "상승" if usd_delta > 0 else "하락"
        sign   = "증가" if usd_delta > 0 else "감소"
        st.markdown(f"- USD 환율 전월 대비 @{usd_delta:,.1f}원 {updown}으로 영업이익 {effect:,.2f}억 {sign}")

    except Exception as e:
        st.error(f"수출 환율 차이 생성 중 오류: {e}")

with t3:

    st.markdown("<h4>1) 포스코 對 JFE 입고가격 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)

    try:
        file_name = st.secrets["sheets"]["f_23"]  # 파일 경로/시크릿 키는 환경에 맞게
        df_src = pd.read_csv(file_name, dtype=str)

        # 숫자필드는 모듈에서 처리하므로 여기서는 최소 정리만
        df_src["연도"] = pd.to_numeric(df_src["연도"], errors="coerce")
        df_src["월"]   = pd.to_numeric(df_src["월"],   errors="coerce")

        sel_y = int(st.session_state["year"])
        sel_m = int(st.session_state["month"])

        wide, col_order, hdr1_labels, hdr2_labels = modules.build_posco_jfe_price_wide(
            df_src, sel_y, sel_m,
            group_name="포스코 對 JFE 입고가격",
            monthly_years=(2021, 2022, 2023, 2024)
        )

        # === 표시용 변환: 문자열은 그대로, NaN만 빈칸 ===
        vis = wide.copy()
        for c in vis.columns:
            vis[c] = [("" if (isinstance(x, float) and pd.isna(x)) else x) for x in vis[c]]

        # 인덱스 → 컬럼
        disp = vis.reset_index()
        disp.rename(columns={"kind":"구분","party":"세부","item":"항목"}, inplace=True)

        # 스페이서(첫 칸)
        SPACER = "__spacer__"
        disp.insert(0, SPACER, "")

        cols = disp.columns.tolist()
        data_cols = [c for c in cols if c not in (SPACER, "구분", "세부", "항목")]

        # 2행 가짜 헤더
        dyn_pat = re.compile(r"^(?P<m>\d{1,2})월\((?P<y>\d{4})\)$")

        hdr1 = ["", "", "", ""]
        for c in data_cols:
            if c.endswith("년 월평균"):
                hdr1.append(c[:4] + "년")
            else:
                m = dyn_pat.match(c)
                hdr1.append(f"{sel_y}년" if m else "")

        hdr2 = ["", "", "구분", ""]
        for c in data_cols:
            if c.endswith("년 월평균"):
                hdr2.append("월평균")
            else:
                m = dyn_pat.match(c)
                hdr2.append(f"{int(m.group('m'))}월" if m else "")

        hdr_df  = pd.DataFrame([hdr1, hdr2], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)

        # ===== 테이블 스타일 =====
        styles = [
            {'selector': 'thead', 'props': [('display','none')]},

            # 가짜 헤더 2행
            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'),('font-weight','700'),('padding','6px 8px')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'),('font-weight','700'),('padding','8px 8px')]},

            # 본문 중앙정렬
            {'selector': 'tbody tr:nth-child(n+3) td', 'props': [('text-align','center')]},

            # 좌측 설명 칸
            {'selector': 'tbody tr td:nth-child(1)', 'props': [('text-align','center'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(2)', 'props': [('text-align','center'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(3)', 'props': [('text-align','center'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(4)', 'props': [('text-align','center'),('white-space','nowrap')]},

            # 헤더 하단 굵은선
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('border-bottom','3px solid gray !important')]},

            # 항목(4번째) 오른쪽 굵은 경계
            {'selector': 'tbody tr:nth-child(n+1) td:nth-child(4)', 'props':[('border-right','3px solid gray !important')]},
        ]

        data_start = 5
        n_cols = len(cols)
        for j in range(data_start, n_cols+1):
            styles += [
                {'selector': f'tbody tr:nth-child(1) td:nth-child({j})', 'props':[('border-top','3px solid gray !important')]},
                {'selector': f'tbody tr:nth-child(n+3) td:nth-child({j})', 'props':[('border-right','2px solid #eee')]},
            ]

        # 필요 시 구분선(섹션 경계)을 추가하고 싶다면 여기서 조건부로 row 찾은 뒤 border-bottom 주입 가능

        # 음수/괄호 붉은색은 사용 안함(변동폭 화살표는 문자열)
        def _noop(_): return ''

        row_labels = disp_vis.index[2:]
        col_labels = disp_vis.columns[(data_start-1):]

        display_styled_df(
            disp_vis,
            styles=styles,
            already_flat=True,
            applymap_rules=[(_noop, (row_labels, col_labels))]
        )

    except Exception as e:
        st.error(f"포스코 對 JFE 입고가격 생성 오류: {e}")

    st.divider()

    st.markdown("<h4>2) 포스코/JFE 투입비중 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)

    try:
        
        file_name = st.secrets["sheets"]["f_24"]
        df_src = pd.read_csv(file_name, dtype=str)  
        df_src["연도"] = pd.to_numeric(df_src["연도"], errors="coerce")
        df_src["월"]   = pd.to_numeric(df_src["월"],   errors="coerce")


        sel_y = int(st.session_state["year"])
        sel_m = int(st.session_state["month"])

        ret = modules.build_posco_jfe_wide(df_src, sel_y, sel_m)
        wide = ret[0] if isinstance(ret, tuple) else ret


        def _fmt(idx, v):
            if pd.isna(v):
                return ""
            metric = idx[2] if isinstance(idx, tuple) and len(idx) > 2 else ""
            if metric == "비중":
                return f"({abs(v):.1f}%)" if v < 0 else f"{v:.1f}%"
            iv = int(round(v))
            return f"({abs(iv):,})" if v < 0 else f"{iv:,}"

        vis = wide.copy()
        for c in vis.columns:
            vis[c] = [_fmt(i, x) for i, x in zip(vis.index, vis[c])]




        def _month_shift(y: int, m: int, delta: int):
            t = y * 12 + (m - 1) + delta
            ny = t // 12
            nm = t % 12 + 1
            return int(ny), int(nm)

        prev2_y, prev2_m = _month_shift(sel_y, sel_m, -2)
        prev_y,  prev_m  = _month_shift(sel_y, sel_m, -1)

        disp = vis.reset_index()
        disp.rename(columns={"kind":"구분","sub":"세부","metric":"항목"}, inplace=True)
        SPACER = "__spacer__"
        disp.insert(0, SPACER, "")

        cols = disp.columns.tolist()
        data_cols = [c for c in cols if c not in (SPACER, "구분", "세부", "항목")]


        dyn_pat = re.compile(r"^(?P<m>\d{1,2})월\((?P<y>\d{4})\)$")


        hdr1 = ["", "", "", ""]
        for c in data_cols:
            if c.endswith("년 월평균"):
                hdr1.append(c[:4] + "년")               
            else:
                m = dyn_pat.match(c)
                if m:
                    hdr1.append(f"{sel_y}년")          
                else:
                    hdr1.append("")


        hdr2 = ["", "", "구분", ""]
        for c in data_cols:
            if c.endswith("년 월평균"):
                hdr2.append("월평균")
            else:
                m = dyn_pat.match(c)
                if m:
                    hdr2.append(f"{int(m.group('m'))}월")  # 실제 월 표시
                else:
                    hdr2.append("")

        hdr_df  = pd.DataFrame([hdr1, hdr2], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)

        styles = [
            {'selector': 'thead', 'props': [('display','none')]},

            # 가짜 헤더 2행
            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'),('font-weight','700'),('padding','6px 8px')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'),('font-weight','700'),('padding','8px 8px')]},

            # 본문 우측정렬
            {'selector': 'tbody tr:nth-child(n+3) td', 'props': [('text-align','center')]},

            # 좌측 설명 칸
            {'selector': 'tbody tr td:nth-child(1)', 'props': [('text-align','center'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(2)', 'props': [('text-align','center'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(3)', 'props': [('text-align','center'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(4)', 'props': [('text-align','center'),('white-space','nowrap')]},

            # 헤더 하단 굵은선
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('border-bottom','3px solid gray !important')]},

            # 항목(4번째) 오른쪽 굵은 경계
            {'selector': 'tbody tr:nth-child(n+1) td:nth-child(4)', 'props':[('border-right','3px solid gray !important')]}
        ]

        data_start = 5
        n_cols = len(cols)
        for j in range(data_start, n_cols+1):
            styles += [
                {'selector': f'tbody tr:nth-child(1) td:nth-child({j})', 'props':[('border-top','3px solid gray !important')]},
                {'selector': f'tbody tr:nth-child(n+3) td:nth-child({j})', 'props':[('border-right','2px solid #eee')]}
            ]

        disp_vis.iloc[0, 8] = "" ; disp_vis.iloc[0, 10] = "" ; 
        disp_vis.iloc[3, 1] = "" ; disp_vis.iloc[3, 2]  = "" ; disp_vis.iloc[4, 1] = "" ;disp_vis.iloc[5, 1] = "" ; disp_vis.iloc[6, 1] = "" ; disp_vis.iloc[9, 1] = "" ; disp_vis.iloc[11, 1] = ""
        disp_vis.iloc[5, 1] = "" ; disp_vis.iloc[5, 2]  = ""
        disp_vis.iloc[8, 1] = "" ; disp_vis.iloc[8, 2]  = ""
        disp_vis.iloc[10, 1] = "" ; disp_vis.iloc[10, 2]  = ""
        disp_vis.iloc[6, 3] = "" ; disp_vis.iloc[11, 3] = ""
        disp_vis.iloc[12, 3] = "" ; disp_vis.iloc[13, 3] = "" 

        spacer_rules1 = [
            {
                'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                'props': [('border-bottom','2px solid white !important')]
               
            }
            for r in (1,2,3,4,5,6,7,8)
        ]

        styles += spacer_rules1

        spacer_rules2 = [
            {
                'selector': f'tbody tr:nth-child({j}) td:nth-child({r})',
                'props': [('border-right','2px solid white !important')]
               
            }
            for r in (1,2,3,9,10)
            for j in (1,2)
        ]

        styles += spacer_rules2

        spacer_rules3 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                'props': [('border-right','2px solid white !important')]
               
            }
            for r in range(3,15)
        ]

        styles += spacer_rules3

        spacer_rules4 = [
            {
                'selector': f'tbody tr:nth-child({r}) ',
                'props': [('border-bottom','3px solid gray !important')]
               
            }
            for r in (7,12)
        ]

        styles += spacer_rules4

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({j}) td:nth-child({r})',
                'props': [('border-bottom','2px solid white !important')]
               
            }
            for r in (1,2)
            for j in range(3,14)
        ]

        styles += spacer_rules5

        spacer_rules6 = [
            {
                'selector': f'tbody tr:nth-child({j}) td:nth-child(3)',
                'props': [('border-bottom','2px solid white !important')]
               
            }

            for j in (3,5,8,10)
        ]

        styles += spacer_rules6



        def _neg_red(val):
            if isinstance(val, str) and val.strip().startswith("("):
                return 'color: #d32f2f;'
            return ''

        row_labels = disp_vis.index[2:]                   
        col_labels = disp_vis.columns[(data_start-1):]    

        display_styled_df(
            disp_vis,
            styles=styles,
            already_flat=True,
            applymap_rules=[(_neg_red, (row_labels, col_labels))]
        )

    except Exception as e:
        st.error(f"포스코/JFE 입고가격 생성 오류: {e}")

    st.divider()

    st.markdown("<h4>3) 메이커별 입고추이 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 톤/천원]</div>", unsafe_allow_html=True)

    import itertools  


    try:
        # 1) 데이터
        file_name = st.secrets["sheets"]["f_25"]
        df_src = pd.read_csv(file_name, dtype=str)
        df_src["연도"] = pd.to_numeric(df_src["연도"], errors="coerce")
        df_src["월"]   = pd.to_numeric(df_src["월"],   errors="coerce")

        # 2) 선택 연월
        sel_y = int(st.session_state["year"])
        sel_m = int(st.session_state["month"])



        wide, cols_mi = modules.build_maker_receipt_wide(df_src, sel_y, sel_m, base_year=sel_y-1)

        def _fmt_number(x):
            if pd.isna(x): 
                return ""
            iv = int(round(float(x)))
            return f"{iv:,}"

        def fmt_cell(idx, col, v):

            if pd.isna(v):
                return ""

            item = idx[1]        # 중량/단가/증감
            lower = col[1]       # 월평균/매입비중/중량

            # 매입비중
            if lower == "매입비중":
                x = float(v)
                return f"{x:.1f}%" if x >= 0 else f"({abs(x):.1f}%)"

            # 중량/월평균: 백의자리 반올림 후 1000으로 축약
            if item == "중량" and lower in ("월평균", "중량"):
                x = modules._thousand_out(float(v))
                if pd.isna(x):
                    return ""
                return _fmt_number(x)
            
            if item == "단가" and lower in ("월평균","중량"):
                x = modules._thousand_out(float(v))
                if pd.isna(x):
                    return ""
                return _fmt_number(x)

            # 증감
            if item == "증감" and lower in ("중량",):
                iv = modules._thousand_out(float(v))
                if iv > 0:
                    return f'<span style="color:#1f77b4;">▲{abs(iv):,}</span>'
                elif iv < 0:
                    return f'<span style="color:#d62728;">▼{abs(iv):,}</span>'
                else:
                    return "0"


            return ""

        body = wide.copy()
        for col in body.columns:
            for idx in body.index:
                body.at[idx, col] = fmt_cell(idx, col, body.at[idx, col])

        SPACER = "__spacer__"
        disp = body.reset_index()
        disp.insert(0, SPACER, "")
        cols = disp.columns.tolist()

        hdr1 = ["", "구분", "항목"] + [c[0] for c in cols_mi]
        hdr2 = ["", "구분", "항목"] + [c[1] for c in cols_mi]

        hdr_df   = pd.DataFrame([hdr1, hdr2], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)


        n_fixed = 3 
        top_labels = [c[0] for c in cols_mi]
        group_edges, j = [], n_fixed
        for _, g in groupby(top_labels):
            g_len = len(list(g))
            start = j + 1
            end   = j + g_len
            group_edges.append((start, end))
            j = end

        styles = [
            {'selector': 'thead', 'props': [('display','none')]},

            {'selector': 'tbody tr:nth-child(1) td', 'props': [
                ('text-align','center'),('font-weight','700'),('padding','6px 8px')
            ]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [
                ('text-align','center'),('font-weight','700'),('padding','8px 8px'),
                ('border-bottom','3px solid gray !important')
            ]},

            # 데이터 정렬
            {'selector': 'tbody tr:nth-child(n+3) td', 'props': [('text-align','right')]},
            {'selector': 'tbody tr td:nth-child(1)', 'props': [('text-align','left'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(2)', 'props': [('text-align','left'),('white-space','nowrap')]},
            {'selector': 'tbody tr td:nth-child(3)', 'props': [('text-align','left'),('white-space','nowrap')]},

            {'selector': 'tbody tr:nth-child(n+1) td:nth-child(3)', 'props':[('border-right','3px solid gray !important')]},
        ]

        # 헤더 최상단 라인
        for k in range(n_fixed+1, len(cols)+1):
            styles.append({'selector': f'tbody tr:nth-child(1) td:nth-child({k})',
                        'props':[('border-top','3px solid gray !important')]})

        # 그룹 경계선
        for (_, end) in group_edges:
            styles.append({'selector': f'tbody tr:nth-child(n+1) td:nth-child({end})',
                        'props':[('border-right','3px solid gray !important')]})



        display_styled_df(
            disp_vis,
            styles=styles,
            already_flat=True,
        )


    except Exception as e:
        st.error(f"메이커별 입고추이 표 생성 오류: {e}")

    st.divider()

with t4:

    st.markdown("<h4>1) 제조 가공비 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)



    try:

        file_name = st.secrets["sheets"]["f_26"]  
        df_src = pd.read_csv(file_name, dtype=str)
        df_src["연도"] = pd.to_numeric(df_src["연도"], errors="coerce")
        df_src["월"]   = pd.to_numeric(df_src["월"],   errors="coerce")


        sel_y = int(st.session_state["year"])
        sel_m = int(st.session_state["month"])


        disp_raw, meta = modules.build_mfg_cost_table(df_src, sel_y, sel_m)
        prev_y, prev_m, cur_y, cur_m = meta["prev_y"], meta["prev_m"], meta["sel_y"], meta["sel_m"]


        flat_cols = ["구분"]
        for top in ["전월", "당월", "전월대비"]:
            for sub in ["포항", "충주", "충주2", "계"]:
                flat_cols.append(f"{top}|{sub}")
        disp = disp_raw.copy()
        disp.columns = flat_cols  

        # 2) 스페이서 추가 
        SPACER = "__spacer__"
        disp.insert(0, SPACER, "")
        cols = disp.columns.tolist()  

        hdr1 = ["", ""] \
            + [f"{prev_m}월"] * 4 \
            + [f"{cur_m}월"] * 4 \
            + ["전월대비"] * 4
        hdr2 = ["", "구분"] + (["포항","충주","충주2","계"] * 3)

        hdr_df   = pd.DataFrame([hdr1, hdr2], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)
 
        def fmt_num(v):
            if pd.isna(v): 
                return ""
            iv = int(round(float(v)))
            return f"({abs(iv):,})" if iv < 0 else f"{iv:,}"

        def fmt_cell(key, v):

            if "|" not in key:
                return v
            if pd.isna(v):
                return ""
            iv = int(round(float(v)))
            top, _ = key.split("|", 1)
            if top == "전월대비":
                if iv > 0:  return f'<span style="color:#000000;">{abs(iv):,}</span>' 
                if iv < 0:  return f'<span style="color:red;">({abs(iv):,})</span>'  
                return "0"
            return fmt_num(iv)

        body = disp_vis.copy()


        data_rows = body.index[2:]  
        for c in body.columns[2:]:  
            body.loc[data_rows, c] = body.loc[data_rows, c].apply(lambda x, kk=c: fmt_cell(kk, x))

        styles = [
            {'selector': 'thead', 'props': [('display','none')]},
            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'),('font-weight','700')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'),('font-weight','700'),('border-bottom','3px solid gray !important')]},
            {'selector': 'tbody tr:nth-child(n+3) td', 'props': [('text-align','right')]},
            {'selector': 'tbody tr td:nth-child(2)', 'props': [('text-align','left'),('white-space','nowrap'),('border-right','3px solid gray !important')]},
        ]

        # 헤더 윗선
        for k in range(3, len(cols)+1):
            styles.append({'selector': f'tbody tr:nth-child(1) td:nth-child({k})',
                        'props':[('border-top','3px solid gray !important')]})

        display_styled_df(body, styles=styles, already_flat=True)


    except Exception as e:
        st.error(f"제조 가공비 표 생성 오류: {e}")

    st.divider()

with t5:
    st.markdown("<h4>1) 판매비와 관리비 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원]</div>", unsafe_allow_html=True)

    try:
        # 1) 데이터
        file_name = st.secrets["sheets"]["f_27"]
        df_src = pd.read_csv(file_name, dtype=str)
        df_src = pd.read_csv(file_name, dtype=str)
        df_src["연도"] = pd.to_numeric(df_src["연도"], errors="coerce")



        disp_raw, meta = modules.build_sgna_table(df_src, sel_y, sel_m)
        avg_years = meta.get("avg_years", [])   
        m2, m1, m0 = meta["months"]

       
        avg_cols = [f"'{y}년 월평균" for y in avg_years]
        desired = ["구분"] + avg_cols + [f"{m2}월", f"{m1}월", f"{m0}월", "전월대비"]

        desired = [c for c in desired if c in disp_raw.columns]
        disp = disp_raw[desired].copy()

        SPACER="__sp__"
        disp.insert(0, SPACER, "")
        cols = disp.columns.tolist()

        hdr1 = ["", ""] + [f"'{y}년" for y in avg_years]
        while len(hdr1) < len(cols): hdr1.append("")
        hdr2 = ["", "구분"] + ["월평균"]*len(avg_years)
        while len(hdr2) < len(cols): hdr2.append(cols[len(hdr2)])

        hdr_df   = pd.DataFrame([hdr1, hdr2], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)



        # ====== 숫자 포맷(데이터 행만 적용) ======
        def fmt_num(v):
            if pd.isna(v): return ""
            iv = int(round(float(v)))
            return f"({abs(iv):,})" if iv < 0 else f"{iv:,}"

        def fmt_diff(v):
            if pd.isna(v): return ""
            iv = int(round(float(v)))
            if iv < 0: return f'<span style="color:#d62728;">({abs(iv):,})</span>'
            if iv > 0: return f"{iv:,}"
            return "0"

        body = disp_vis.copy()
        data_rows = body.index[2:]  
        for c in body.columns[2:]:  
            if c == "전월대비":
                body.loc[data_rows, c] = body.loc[data_rows, c].apply(fmt_diff)
            else:
                body.loc[data_rows, c] = body.loc[data_rows, c].apply(fmt_num)

     
        styles = [
            {'selector': 'thead', 'props': [('display','none')]},
            {'selector': 'tbody tr:nth-child(1) td', 'props': [
                ('text-align','center'),('font-weight','700'),('border-bottom','2px solid #000 !important')
            ]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [
                ('text-align','center'),('font-weight','700')
            ]},
            {'selector': 'tbody tr:nth-child(n+3) td:nth-child(2)', 'props': [('text-align','left'),('white-space','nowrap')]},
            {'selector': 'tbody tr:nth-child(n+3) td:nth-child(n+3)', 'props': [('text-align','right')]},
            {'selector': 'tbody tr td:nth-child(2)', 'props': [('border-right','3px solid gray !important')]},  # 구분 경계
        ]


        display_styled_df(body, styles=styles, already_flat=True)





    except Exception as e:
        st.error(f"판매비와 관리비 표 생성 오류: {e}")

    st.divider()




with t6:
    st.markdown("<h4>1) 성과급 및 격려금 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 백만원]</div>", unsafe_allow_html=True)

    try:
        file_name = st.secrets["sheets"]["f_28"]
        df_src = pd.read_csv(file_name, dtype=str)  # 숫자 변환은 모듈에서 처리

        sel_y = int(st.session_state["year"])
        sel_m = int(st.session_state["month"])

        disp_raw, meta = modules.build_bonus_table_28(df_src, sel_y, sel_m)
        ytd_lbl = meta["ytd_lbl"]
        
        # === 헤더 2줄 ===
        SPACER="__sp__"
        disp = disp_raw.copy()
        disp.insert(0, SPACER, "")
        cols = disp.columns.tolist()

        hdr1 = ["","", "당월","당월","당월", ytd_lbl, ytd_lbl, ytd_lbl, "100% 금액", "100% 금액"]
        hdr2 = ["","구분","계획","실적","차이","계획","실적","차이","연간","월"]

        hdr_df   = pd.DataFrame([hdr1, hdr2], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)

        # === 숫자 포맷 ===
        def fmt_num(v):
            if pd.isna(v): return ""
            iv = modules._thousand_out(round(float(v)))
            return f"({abs(iv):,})" if iv < 0 else f"{iv:,}"
        
        

        def fmt_diff(v):
            if pd.isna(v): return ""
            iv = modules._thousand_out(round(float(v)))
            if iv < 0: return f'<span style="color:#d62728;">({abs(iv):,})</span>'
            if iv > 0: return f"{iv:,}"
            return "0"

        body = disp_vis.copy()
        data_rows = body.index[2:]
        diff_cols = [c for c in cols if c.endswith("|차이")]
        for c in body.columns[2:]:
            if c in diff_cols:
                body.loc[data_rows, c] = body.loc[data_rows, c].apply(fmt_diff)
            else:
                body.loc[data_rows, c] = body.loc[data_rows, c].apply(fmt_num)

        # === 스타일 ===
        styles = [
            {'selector': 'thead', 'props': [('display','none')]},
            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'),('font-weight','700')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'),('font-weight','700'),('border-bottom','3px solid #000 !important')]},
            {'selector': 'tbody tr:nth-child(n+3) td:nth-child(2)', 'props': [('text-align','left'),('white-space','nowrap')]},
            {'selector': 'tbody tr:nth-child(n+3) td:nth-child(n+3)', 'props': [('text-align','right')]},

            {'selector': 'tbody tr td:nth-child(2)', 'props': [('border-right','3px solid #000 !important')]},
        ]


        display_styled_df(body, styles=styles, already_flat=True)

    except Exception as e:
        st.error(f"성과급 및 격려금 표 생성 오류: {e}")

    st.divider()


with t7:
    st.markdown("<h4>1) 통상임금 </h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 백만원]</div>",unsafe_allow_html=True)

    try:
        file_name = st.secrets["sheets"]["f_29"]
        df_src = pd.read_csv(file_name, dtype=str)

        sel_y = int(st.session_state["year"])

        disp_raw = modules.build_wage_table_29(df_src, sel_y)


        SPACER = "__sp__"
        disp = disp_raw.copy()


        insert_pos = disp.columns.get_loc("항목") + 1
        disp.insert(insert_pos, SPACER, "")

        cols = disp.columns.tolist()

        # 헤더도 동일한 순서로 8개

        hdr = ["구분", "항목", "", "1분기", "2분기", "3분기", "4분기", "연간"]
        hdr_df = pd.DataFrame([hdr], columns=cols)

        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)

        # 숫자 포맷
        def fmt_num(v):
            if pd.isna(v):
                return ""
            iv = modules._thousand_out(round(float(v)))
            return f"{iv:,}"

        body = disp_vis.copy()
        data_rows = body.index[1:]  # 첫 행은 헤더

        num_cols = ["1분기", "2분기", "3분기", "4분기", "연간"]
        for c in num_cols:
            if c in body.columns:
                body.loc[data_rows, c] = body.loc[data_rows, c].apply(fmt_num)

        # 스타일
        styles = [
            # 기본 thead 숨김
            {'selector': 'thead', 'props': [('display', 'none')]},


            {
                'selector': 'tbody tr:nth-child(1) td',
                'props': [
                    ('text-align', 'center'),
                    ('font-weight', '700'),
                    ('border-bottom', '3px solid #000 !important')
                ]
            },

            {
                'selector': 'tbody tr:nth-child(n+2) td:nth-child(1),'
                            'tbody tr:nth-child(n+2) td:nth-child(2)',
                'props': [
                    ('text-align', 'left'),
                    ('white-space', 'nowrap')
                ]
            },

            {
                'selector': 'tbody tr:nth-child(n+2) td:nth-child(n+4)',
                'props': [
                    ('text-align', 'right')
                ]
            },

            {
                'selector': 'tbody tr td:nth-child(3)',
                'props': [
                    ('border-right', '3px solid #000 !important')
                ]
            },
        ]

        display_styled_df(body, styles=styles, already_flat=True)

    except Exception as e:
        st.error(f"통상임금 표 생성 오류: {e}")

    st.divider()






# Footer
st.markdown("""
<style>.footer { bottom: 0; left: 0; right: 0; padding: 8px; text-align: center; font-size: 13px; color: #666666;}</style>
<div class="footer">ⓒ 2025 SeAH Special Steel Corp. All rights reserved.</div>
""", unsafe_allow_html=True)