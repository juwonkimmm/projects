import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import modules  

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# =========================
# 공통 테이블 렌더 (인덱스 숨김 + 중복 컬럼 안전)
# =========================
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

def display_styled_df(df, styles=None, highlight_cols=None, already_flat=False):
    """
    - already_flat=True: df가 이미 index 없는 평평한 형태(= reset_index 완료)라고 가정
    """
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
        .format(lambda x: f"{x:,.0f}" if isinstance(x, (int,float,np.integer,np.floating)) and pd.notnull(x) else x)
        .set_properties(**{'text-align':'right','font-family':'Noto Sans KR'})
        .apply(highlight_columns, axis=0)
        .hide(axis="index")  # 👈 인덱스 완전 숨김
    )
    if styles:
        styled_df = styled_df.set_table_styles(styles)

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

st.markdown(f"## {year}년 {month}월 실적 분석")
t1, t2, t3 = st.tabs(['주요경영지표', '주요경영지표(본사)', '연간사업계획'])
st.divider()

# =========================
# 주요경영지표
# =========================

with t1:
    st.markdown("<h4>1) 손익 </h4>", unsafe_allow_html=True)
    st.markdown("<h5> 손익 연결</h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 톤, 백만원, %]</div>", unsafe_allow_html=True)

    try:
        file_name = st.secrets["sheets"]["f_1"]
        df_src = pd.read_csv(file_name)

        snap = modules.create_connected_profit_snapshot_table(
            year=int(st.session_state['year']),
            month=int(st.session_state['month']),
            data=df_src
        )

        # 화면용: '구분' 컬럼 추가(두 %를 모두 '%'로 표시), 인덱스는 제거
        snap_disp = snap.copy()
        snap_disp.insert(0, '구분', snap_disp.index.map(lambda x: '%' if str(x).startswith('%') else x))
        snap_disp = snap_disp.reset_index(drop=True)
        



        highlight_cols = ['전월 실적', '당월 계획', '당월 실적', '전월 실적 대비', '계획 대비']

        styles = (
            {'selector': 'thead th', 'props': [('padding','14px 10px'), ('line-height','2')]},  # 전체 헤더 기본(높음)

            )
        
        


        







        display_styled_df(snap_disp, styles=styles,already_flat=True, highlight_cols=highlight_cols)



        


        st.caption("각 %는 계산")

    except Exception as e:
        st.error(f"손익 연결 생성 중 오류: {e}")

    

    st.divider()
     
    ##### no2 현금흐름표 #####

    st.markdown("<h4>2) 현금흐름표</h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 백만원]</div>", unsafe_allow_html=True)

    try:
        file_name = st.secrets["sheets"]["f_2"]  
        raw = pd.read_csv(file_name, dtype=str)

        # ─ 연산(구분 기준) ─
        base = modules.create_cashflow_snapshot_by_gubun(
            year=int(st.session_state['year']),
            month=int(st.session_state['month']),
            data=raw
        )  # index='구분', cols=['\'24','\'25','당월','본사','남통','천진','태국','당월누적']

        # ─ 표시용 포맷: 천단위, 음수 괄호 ─
        def fmt_cell(x):
            try:
                v = float(x)
            except Exception:
                return x
            return f"({abs(int(round(v))):,})" if v < 0 else f"{int(round(v)):,}"

        disp = base.copy()
        for c in disp.columns:
            disp[c] = disp[c].apply(fmt_cell)
        





        # ─ 구분을 첫 컬럼으로 승격 ─
        disp = base.copy().reset_index()   
        # ─ 스페이서 컬럼을 2번째 위치에 추가 ─
        SPACER_COL = "__spacer__"        
        disp.insert(0, SPACER_COL, "")


        # ─ 3단 헤더 구성 ─
        yy = str(int(st.session_state['year']))[-2:]
        mm = int(st.session_state['month'])
        top = f"'{yy} {mm}월"


        cols = disp.columns.tolist()
        c_idx = {c: i for i, c in enumerate(cols)}   # 라벨→0-based

        month_i = c_idx['당월']
        acc_i   = c_idx['당월누적']
        prev_col = next((c for c in cols if c.startswith("'") and c != "'24"), None)

        yy = str(int(st.session_state['year']))[-2:]
        mm = int(st.session_state['month'])
        top_label = f"'{yy} {mm}월"
        prev_text = f"'{yy} {mm-1}월" if mm > 1 else f"'{yy} 0월"

        hdr1 = [''] * len(cols)
        hdr2 = [''] * len(cols)
        hdr3 = [''] * len(cols)

        # 1행: 상단 그룹/누적
        hdr1[month_i] = top_label
        

        # 2행: 좌측 표제 + 당월 + 전월누계
        hdr2[c_idx['구분']] = '구분'
        hdr2[c_idx["'24"]]  = "'24"
        if prev_col is not None:
            hdr2[c_idx[prev_col]] = prev_text
        hdr2[month_i] = '당월'
        hdr2[acc_i]   = '당월누적'

        # 3행: 회사 라벨
        for k in ['본사','남통','천진','태국']:
            if k in c_idx:
                hdr3[c_idx[k]] = k

        hdr_df   = pd.DataFrame([hdr1, hdr2, hdr3], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)


        # 회사 마지막 열 위치(경계선용)
        company_idxs = [c_idx[k] for k in ['본사','남통','천진','태국'] if k in c_idx]
        last_company_i = max(company_idxs) if company_idxs else month_i

        # ── CSS ──
        styles = [
            # 원래 thead 숨김
            {'selector': 'thead', 'props': [('display','none')]},

            # 헤더 1·2·3행
            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'), ('padding','8px 8px'),  ('line-height','1'), ('font-weight','600')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'), ('padding','10px 8px'), ('line-height','1.5'), ('font-weight','600')]},
            {'selector': 'tbody tr:nth-child(3) td', 'props': [('text-align','center'), ('padding','14px 10px'), ('line-height','0.1'), ('font-weight','600')]},

            # 스페이서 열 전체(모든 행)
            {'selector': 'tbody td:nth-child(1)', 'props': [('width','8px'), ('border-right','0')]},
            


            {'selector': 'tbody td:nth-child(4) td:nth-child(1)',
            'props': [('border-top','3px solid gray !important')]},


            {'selector': 'tbody td:nth-child(1)',
            'props': [('border-right','2px solid white !important')]},




            

        ]

        # 구분 내 항목 왼쪽 정렬
        spacer_rules1 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('text-align','left')]
               
            }
            for r in (4,5,6,9,15,16,17,20,24,25,26,27)
        ]

        styles += spacer_rules1

        #구분 내 항목 구분
        spacer_rules2 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                'props': [('border-right','3px solid gray !important')]
               
            }
            for r in (5,6,7,8,9,10,11,12,13,14,15,16,18,19,21,22,23)
        ]

        styles += spacer_rules2

        #추가 열 공백 구분
        spacer_rules2_1 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                'props': [('border-bottom','2px solid white !important')]
               
            }
            for r in (4,5,6,7,8,9,10,11,12,13,14,15,17,18,20,21,22)
            # for r in (5,6)
        ]

        styles += spacer_rules2_1
        
        #구분 상단 & 하단 검은 선 구분
        spacer_rules3 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-top','3px solid gray !important')]
               
            }

            for r in (4,5,6,9,15,16,17,18,19,20,21,24,25,26,27)
        ]

        styles += spacer_rules3
        
        spacer_rules3_1 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                'props': [('border-top','3px solid gray !important')]
               
            }

            for r in (4,17,20,24,25,26,27)
        ]

        styles += spacer_rules3_1



        # 구분 내 소그룹 그분
        spacer_rules4 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                'props': [('border-top','2px solid white !important')]
               
            }

            for r in (7,8,10,11,12,13,14,19,22,23)
        ]

        styles += spacer_rules4

        ####feature 구분####

        #행 구분

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child({j})',
                'props': [('border-top','2px solid white ')]
               
            }
            # for r in (4,5,8,14,15)
            for r in (2,3)
            for j in (1,2,3,4)
        ]


        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({j})',
                'props': [('border-top','3px solid gray ')]
               
            }

            for j in (5,6,7,8,9,10)
        ]
        

        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(3)',
                'props': [('border-left','3px solid gray ')]
               
            }

            for r in range(4,28)
        ]
        

        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(3) td:nth-child({j})',
                'props': [('border-top','3px solid gray ')]
               
            }

            for j in range (6,10)
        ]

        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(4) td:nth-child({j})',
                'props': [('border-top','3px solid gray ')]
               
            }

            for j in range(3,11)
        ]
        

        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child({j})',
                'props': [('border-right','3px solid gray ')]
               
            }

            for r in range (1,4)
            for j in range (2,5)
        ]
        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(1) td:nth-child(5)',
                'props': [('border-right','3px solid gray ')]
               
            }

        ]
        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(3) td:nth-child({j})',
                'props': [('border-top','2px solid white ')]
               
            }

            for j in (5,10)
        ]
        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({j}) td:nth-child(10)',
                'props': [('border-right','3px solid gray ')]
               
            }

            for j in (1,2,3)
        ]
        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child(5)',
                'props': [('border-right','3px solid gray ')],
                
            }



        ]
        styles += spacer_rules5

        spacer_rules5 = [
            {
                
                'selector': f'tbody tr:nth-child(2) td:nth-child(10)',
                'props': [('border-left','3px solid gray ')]
               
            }



        ]
        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({j})',
                'props': [('border-right','2px solid white ')],
                
            }

            for j in range(6,9)



        ]
        styles += spacer_rules5





        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(1) td:nth-child({j})',
                'props': [('border-right','2px solid white ')],
                
            }

            for j in range(6,10)



        ]
        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({j})',
                'props': [('border-under','2px solid white !important')],
                
            }

            for j in range(5,6)



        ]
        styles += spacer_rules5


        







        display_styled_df(
            disp_vis,
            styles=styles,
            # highlight_cols=highlight_cols,
            already_flat=True
        )

    except Exception as e:
        st.error(f"현금흐름표 생성 중 오류: {e}")

    st.divider()


    ######재무상태표


    st.markdown("<h4>3) 재무상태표</h4>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:right; font-size:13px; color:#666;'>[단위: 백만원]</div>", unsafe_allow_html=True)

    try:
        # 데이터 로드
        file_name = st.secrets["sheets"]["f_3"]
        raw = pd.read_csv(file_name, dtype=str)

        # 모듈 갱신(수정 반영)
        import importlib
        importlib.invalidate_caches(); importlib.reload(modules)

        # 원하는 행 순서(=구분3 값)
        item_order = [
            '현금및현금성자산','매출채권','재고자산','유형자산','기타','자산총계',
            '매입채무','차입금','기타','부채총계',
            '자본금','이익잉여금','기타','자본총계','부채 및 자본 총계'
        ]

        # ─ 연산: 구분3만으로 집계 ─
        base = modules.create_bs_snapshot_by_items(
            year=int(st.session_state['year']),
            month=int(st.session_state['month']),
            data=raw,
            item_order=item_order
        )

        # ─ 표시용 숫자 포맷 ─
        def fmt_cell(x):
            if pd.isna(x): 
                return ""
            try:
                v = float(x)
            except Exception:
                return x
            return f"({abs(int(round(v))):,})" if v < 0 else f"{int(round(v)):,}"

        disp = base.copy().fillna(0)
        for c in disp.columns:
            disp[c] = disp[c].apply(fmt_cell)

        # ─ 구분을 2열로: 스페이서 컬럼 추가 ─
        disp = disp.reset_index()   # '구분' 컬럼 생성
        SPACER = "__spacer__"
        disp.insert(0, SPACER, "")

        # ─ 3단 헤더(가짜 헤더 3행 삽입) ─
        cols = disp.columns.tolist()
        c_idx = {c:i for i,c in enumerate(cols)}
        gu_i    = c_idx['구분']
        month_i = c_idx['당월']
        diff_i  = c_idx['전월비 증감']

        yy = str(int(st.session_state['year']))[-2:]
        used_m = base.attrs.get('used_month', int(st.session_state['month']))
        prev_m = base.attrs.get('prev_month', max(1, used_m-1))
        top_label  = f"'{yy} {used_m}월"
        prev_text  = f"'{yy} {prev_m}월"

        company_labels = [c for c in cols if c not in [SPACER,'구분',"'24","'25",'당월','전월비 증감']]

        hdr1 = [''] * len(cols); hdr1[month_i] = top_label; 
        hdr2 = [''] * len(cols); hdr2[gu_i] = '구분'; hdr2[c_idx["'24"]] = "'24"; hdr2[c_idx["'25"]] = prev_text; hdr2[month_i] = '당월'; hdr2[diff_i] = '전월비 증감'
        hdr3 = [''] * len(cols); 
        for k in company_labels: hdr3[c_idx[k]] = k

        hdr_df   = pd.DataFrame([hdr1, hdr2, hdr3], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)

        last_company_i = max((c_idx[k] for k in company_labels), default=month_i)

        # ─ CSS ─
        styles = [
            {'selector': 'thead', 'props': [('display','none')]},

            # 헤더 1·2·3행
            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'), ('padding','8px 8px'),  ('line-height','1.1'), ('font-weight','600')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'), ('padding','10px 8px'), ('line-height','1.4'), ('font-weight','600')]},
            {'selector': 'tbody tr:nth-child(2) td:nth-child(2)', 'props': [('text-align','center')]},
            {'selector': 'tbody tr:nth-child(3) td', 'props': [('text-align','center'), ('padding','14px 10px'), ('line-height','1.7'), ('font-weight','600')]},

            # 1열 얇게
            {'selector': 'tbody td:nth-child(1)', 'props': [('width','8px'), ('border-right','0')]},


            # 본문
            {'selector': 'tbody tr:nth-child(n+4) td', 'props': [('line-height','1.45'), ('padding','8px 10px'), ('text-align','right')]},
            {'selector': 'tbody tr:nth-child(n+4) td:nth-child(2)', 'props': [('text-align','left')]},
        ]

        spacer_rules1 = [
                    {
                        'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                        'props': [('text-align','left')]
                    
                    }
                    for r in range(4,19)
                ]
        
        styles += spacer_rules1
        

        spacer_rules2 = [
                    {
                        'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                        'props': [('border-left','3px solid gray ')],
                    
                    }
                    for r in (4,5,6,7,8,10,11,12,14,15,16)
                ]
        
        
        styles  += spacer_rules2

        spacer_rules2 = [
                    {
                        'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                        'props': [('border-left','2px solid white ')],
                    
                    }
                    for r in (9,13,17,18)
                ]
        
        
        styles  += spacer_rules2

        spacer_rules3 = [
                    {
                        'selector': f'tbody tr:nth-child({r}) td:nth-child(2)',
                        'props': [('border-top','3px solid gray ')],
                    
                    }
                    for r in (9,10,13,14,17,18)
                ]
        
        styles  += spacer_rules3

        spacer_rules4 = [
                    {
                        'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                        'props': [('border-bottom','2px solid white ')],
                    
                    }
                    for r in range(4,18)
                ]
        
        styles  += spacer_rules4

        spacer_rules5 = [
                    {
                        'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                        'props': [('border-bottom','3px solid gray ')],
                    
                    }
                    for r in (3,9,13,17)
                ]
        
        styles  += spacer_rules5

        spacer_rules5 = [
                    {
                        'selector': f'tbody tr:nth-child(3) td:nth-child(2)',
                        'props': [('border-bottom','3px solid gray ')],
                    
                    }

                ]
        
        styles  += spacer_rules5        

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child(3)',
                'props': [('border-left','3px solid gray ')]
               
            }

            for r in range(4,19)
        ]
        

        styles += spacer_rules5

        ####feature 구분####

        #행 구분

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child({j})',
                'props': [('border-top','2px solid white ')]
               
            }
            # for r in (4,5,8,14,15)
            for r in (2,3)
            for j in (1,2,3,4)
        ]


        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({j})',
                'props': [('border-top','3px solid gray ')]
               
            }

            for j in (5,6,7,8,9,10)
        ]
        

        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(3) td:nth-child({j})',
                'props': [('border-top','3px solid gray ')]
               
            }

            for j in range (6,10)
        ]

        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(4) td:nth-child({j})',
                'props': [('border-top','3px solid gray ')]
               
            }
            # for r in (4,5,8,14,15)
            # for r in (2)
            for j in range(3,11)
        ]
        

        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({r}) td:nth-child({j})',
                'props': [('border-right','3px solid gray ')]
               
            }

            for r in range (1,4)
            for j in range (2,5)
        ]
        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(1) td:nth-child(5)',
                'props': [('border-right','3px solid gray ')]
               
            }

        ]
        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(3) td:nth-child({j})',
                'props': [('border-top','2px solid white ')]
               
            }

            for j in (5,10)
        ]
        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child({j}) td:nth-child(10)',
                'props': [('border-right','3px solid gray ')]
               
            }

            for j in (1,2,3)
        ]
        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child(5)',
                'props': [('border-right','3px solid gray ')],
                
            }



        ]
        styles += spacer_rules5

        spacer_rules5 = [
            {
                
                'selector': f'tbody tr:nth-child(2) td:nth-child(10)',
                'props': [('border-left','3px solid gray ')]
               
            }



        ]
        styles += spacer_rules5

        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({j})',
                'props': [('border-right','2px solid white ')],
                
            }

            for j in range(6,9)



        ]
        styles += spacer_rules5





        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(1) td:nth-child({j})',
                'props': [('border-right','2px solid white ')],
                
            }

            for j in range(6,10)



        ]
        styles += spacer_rules5


        spacer_rules5 = [
            {
                'selector': f'tbody tr:nth-child(2) td:nth-child({j})',
                'props': [('border-under','2px solid white !important')],
                
            }

            for j in range(5,6)



        ]
        styles += spacer_rules5

        spacer_rules10 = [
            {
                
                'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                'props': [('border-right','2px solid white ')],

               
            }
            for r in range (1,4)


        ]
        styles += spacer_rules10


        






        display_styled_df(
            disp_vis,
            styles=styles,
            already_flat=True
        )

    except Exception as e:
        st.error(f"재무상태표 생성 중 오류: {e}")
##

    st.markdown("<h4>4) 회전일</h4>", unsafe_allow_html=True)

    try:
        file_name = st.secrets["sheets"]["f_4"]   # secrets.toml에 f_4 등록
        raw = pd.read_csv(file_name, dtype=str)

        # 최신 modules 반영
        import importlib
        importlib.invalidate_caches(); importlib.reload(modules)

        snap = modules.create_turnover_snapshot(
            year=int(st.session_state['year']),
            month=int(st.session_state['month']),
            data=raw
        )  # MultiIndex cols: ('당월'|'전월비') x (계/회사들)

        # ─ 표시용 포맷: 소수1자리, 음수는 그대로(괄호 X) ─
        def fmt1(x):
            try:
                v = float(x)
            except Exception:
                return x
            return f"{v:.1f}" if pd.notnull(v) else ""

        disp = snap.copy().applymap(fmt1)

        # ─ 가짜 2단 헤더(thead 숨기고 tbody 상단 2행으로 만듦) ─
        disp = disp.reset_index()              # '구분' 컬럼
        SP = "__spacer__"
        disp.insert(0, SP, "")                 # 스페이서 → 1열

        cols = disp.columns.tolist()
        c_idx = {c:i for i,c in enumerate(cols)}

        # 상단 라벨
        yy = str(int(st.session_state['year']))[-2:]
        used_m = snap.attrs.get('used_month', int(st.session_state['month']))
        prev_m = snap.attrs.get('prev_month', max(1, used_m-1))

        # 당월 그룹 시작 끝, 전월비 그룹 시작 끝 계산
        # ('당월','계') 같은 멀티컬럼이 풀린 상태라 라벨을 찾아야 함
        subcols = [c for c in cols if isinstance(c, tuple)]
        # 하지만 reset_index로 풀렸으니, snap에서 직접 사용
        sub_order = list(snap.columns.get_level_values(1).unique())
        left_group_start = 2  # 스페이서, 구분 다음부터
        left_group_end   = left_group_start + len(sub_order) - 1
        right_group_start = left_group_end + 1
        right_group_end   = right_group_start + len(sub_order) - 1

        # 1행:   ['', '',  '당월', '', '', '',  '전월비', '', '', '' ]
        # 2행:   ['', '구분',  '계','특수강',..., '계','특수강',...]
        hdr1 = [''] * len(cols)
        hdr2 = [''] * len(cols)

        hdr1[left_group_start]  = f"'{yy} {used_m}월"
        hdr1[right_group_start] = "전월비"

        hdr2[1] = '구분'  # 2열(스페이서 다음) 구분 표시
        # 하부 소제목 채우기
        for j, name in enumerate(sub_order):
            hdr2[left_group_start + j] = name
            hdr2[right_group_start + j] = name

        hdr_df   = pd.DataFrame([hdr1, hdr2], columns=cols)
        disp_vis = pd.concat([hdr_df, disp], ignore_index=True)

        

        
        def css_overlay_text(r, c, text, strong=True):
        # TD를 기준 위치로
            base = {
                'selector': f'tbody tr:nth-child({r}) td:nth-child({c})',
                'props': [('position', 'relative')],
            }
            # 그 위에 텍스트 올리기
            overlay = {
                'selector': f'tbody tr:nth-child({r}) td:nth-child({c})::after',
                'props': [
                    ('content', f'"{text}"'),
                    ('position', 'absolute'), ('left', '50%'), ('top', '50%'),
                    ('transform', 'translate(-50%, -50%)'),
                    ('white-space', 'nowrap'),
                    ('background', 'transparent'),
                    ('font-weight', '400' if strong else 'normal'),
                ],
            }
            return [base, overlay]

        # 예: 2행 10열에 '전월비'라는 글자를 강제로 표시
        styles = []
        # ─ 스타일 ─
        styles = [
            {'selector': 'thead', 'props': [('display','none')]},

            # 헤더 두 행
            {'selector': 'tbody tr:nth-child(1) td', 'props': [('text-align','center'), ('padding','8px 8px'), ('font-weight','600')]},
            {'selector': 'tbody tr:nth-child(2) td', 'props': [('text-align','center'), ('padding','10px 8px'), ('font-weight','600')]},
            {'selector': 'tbody tr:nth-child(2) td:nth-child(2)', 'props': [('text-align','center')]},

            # 스페이서(1열)
            {'selector': 'tbody td:nth-child(1)', 'props': [('width','8px'), ('border-right','0')]},

            # 본문(3행 이후)
            {'selector': 'tbody tr:nth-child(n+3) td', 'props': [('text-align','right'), ('padding','8px 10px')]},
            {'selector': 'tbody tr:nth-child(n+3) td:nth-child(2)', 'props': [('text-align','center')]},
        ]
        styles += css_overlay_text(3, 1, "회")
        styles += css_overlay_text(4, 1, "전")
        styles += css_overlay_text(5, 1, "일")



        spacer_rules1 = [
                    {
                        'selector': f'tbody tr:nth-child({r}) td:nth-child(1)',
                        'props': [('border-bottom','2px solid white ')],
                    
                    }
                    for r in (1,3,4,5)
                ]
        
        styles  += spacer_rules1

        spacer_rules2 = [
                    {
                        'selector': f'tbody tr:nth-child(1) td:nth-child(2)',
                        'props': [('border-bottom','2px solid white ')],
                    
                    }
                    # for r in (1,3,4,5)
                ]
        
        styles  += spacer_rules2

        spacer_rules3 = [
                    {
                        'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                        'props': [('border-right','2px solid white ')],
                    
                    }
                    for r in (1,4,5,6,9,10,11)
                ]
        
        styles  += spacer_rules3

        spacer_rules4 = [
                    {
                        'selector': f'tbody tr:nth-child(2) td:nth-child(1)',
                        'props': [('border-right','2px solid white ')],
                    
                    }

                ]
        
        styles  += spacer_rules4

        spacer_rules5 = [
                    {
                        'selector': f'tbody tr:nth-child(2) td:nth-child({r})',
                        'props': [('border-top','3px solid gray ')],
                    
                    }
                    for r in (1,2,4,5,6,7,9,10,11,12)
                ]
        
        styles  += spacer_rules5

        spacer_rules6 = [
                    {
                        'selector': f'tbody tr:nth-child(2) td:nth-child({r})',
                        'props': [('border-bottom','3px solid gray ')],
                    
                    }
                    for r in range (1,13)
                ]
        
        styles  += spacer_rules6

        spacer_rules7 = [
                    {
                        'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                        'props': [('border-top','3px solid gray ')],
                    
                    }
                    for r in (3,8)
                ]
        
        styles  += spacer_rules7

        spacer_rules7 = [
                    {
                        'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                        'props': [('border-top','3px solid gray ')],
                    
                    }
                    for r in range(1,13)
                ]
        
        styles  += spacer_rules7

        spacer_rules7 = [
                    {
                        'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                        'props': [('border-left','3px solid gray ')],
                    
                    }
                    for r in (1,3,8,9)
                ]
        
        styles  += spacer_rules7

        spacer_rules7 = [
                    {
                        'selector': f'tbody tr:nth-child(2) td:nth-child({r})',
                        'props': [('border-left','3px solid gray ')],
                    
                    }
                    for r in (1,13)
                ]
        
        styles  += spacer_rules7

        spacer_rules7 = [
                    {
                        'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                        'props': [('border-right','3px solid gray ')],
                    
                    }
                    for r in (3,12)
                ]
        
        styles  += spacer_rules7

        spacer_rules7 = [
                    {
                        'selector': f'tbody tr:nth-child(1) td:nth-child({r})',
                        'props': [('border-right','3px solid gray ')],
                    
                    }
                    for r in (3,12)
                ]
        
        styles  += spacer_rules7

        spacer_rules7 = [
                    {
                        'selector': f'tbody tr:nth-child(2) td:nth-child({r})',
                        'props': [('border-right','3px solid gray ')],
                    
                    }
                    for r in (2,7,12)
 
                ]

        
        styles  += spacer_rules7



        




        display_styled_df(disp_vis, styles=styles, already_flat=True)


    except Exception as e:
        st.error(f"회전일 표 생성 중 오류: {e}")





# =========================
# 주요경영지표(본사)
# =========================
# with t2:
#     pass

# =========================
# 연간사업계획
# =========================
# with t3:
#     pass
# =========================
# Footer
# =========================
st.markdown("""
<style>.footer { bottom: 0; left: 0; right: 0; padding: 8px; text-align: center; font-size: 13px; color: #666666;}</style>
<div class="footer">ⓒ 2025 SeAH Special Steel Corp. All rights reserved.</div>
""", unsafe_allow_html=True)