import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import plotly.graph_objects as go
import modules
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# ===== Korean font setup (Matplotlib + Plotly + CSS) =====
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import plotly.io as pio
import plotly.graph_objects as go

# 1) 리눅스 배포 환경에서 존재 가능성이 높은 경로 후보 등록
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
]

for p in FONT_CANDIDATES:
    if os.path.exists(p):
        fm.fontManager.addfont(p)

# 2) Matplotlib 기본 폰트 패밀리(폴백 포함)
mpl.rcParams["font.family"] = [
    "NanumGothic",          # 리눅스(Cloud)
    "Nanum Barun Gothic",
    "Noto Sans CJK KR",
    "Malgun Gothic",        # Windows
    "AppleGothic",          # macOS
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False

# (선택) 폰트 캐시 강제 리로드
try:
    fm._load_fontmanager(try_read_cache=False)
except Exception:
    pass

# 3) Plotly 기본 템플릿에 한글 폰트 지정(브라우저 폰트 사용)
pio.templates["nanum"] = go.layout.Template(
    layout=dict(font=dict(family="NanumGothic, Noto Sans CJK KR, AppleGothic, Malgun Gothic, sans-serif"))
)
pio.templates.default = "plotly+nanum"

# 4) HTML/테이블용 웹폰트 로드 + CSS 폴백
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap" rel="stylesheet">
<style>
html, body, div, table, th, td {
  font-family: 'Noto Sans KR','NanumGothic','Noto Sans CJK KR', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)
# =========================================================


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


def display_styled_df(df, styles=None, highlight_cols=None):
    """
    - 행 멀티인덱스는 reset_index()로 컬럼 승격 → 왼쪽 숫자 인덱스 제거
    - reset_index()로 생길 수 있는 '중복 컬럼명' 자동 고유화
    - Styler.hide(axis="index")로 인덱스 헤더까지 숨김
    """

    hi_set = set(map(str, (highlight_cols or [])))
    def highlight_columns(col):
        return ['background-color: #f0f0f0'] * len(col) if str(col.name) in hi_set else [''] * len(col)
    
def nudge_texts_to_avoid_overlap(ax, min_sep_px=10):
    """
    같은 x 지점에 있는 텍스트들이 겹치지 않도록 세로로 최소 min_sep_px 픽셀 간격 확보.
    ax 안의 모든 Text 객체를 스캔해서 조정합니다.
    """
    import numpy as np
    import matplotlib.transforms as mtransforms

    # x별 텍스트 분류
    by_x = {}
    texts = [t for t in ax.texts if t.get_visible()]
    for t in texts:
        x, y = t.get_position()
        by_x.setdefault(float(x), []).append(t)

    # 픽셀 ↔ 데이터 변환용 스케일
    ylim = ax.get_ylim()
    height_px = ax.bbox.height if ax.bbox.height > 0 else 1
    data_per_px = (ylim[1] - ylim[0]) / height_px
    min_sep_data = min_sep_px * data_per_px

    # x별로 y 기준 정렬 후 위로 조금씩 띄우기
    for x, ts in by_x.items():
        # y(데이터 좌표) 기준 오름차순 정렬
        ts.sort(key=lambda t: t.get_position()[1])
        last_y = -np.inf
        for t in ts:
            x0, y0 = t.get_position()
            y_new = max(y0, last_y + min_sep_data)  # 최소 간격 확보
            if y_new != y0:
                t.set_position((x0, y_new))
            last_y = y_new

    # 조정 후 다시 그리도록
    ax.figure.canvas.draw_idle()


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
st.image("logo.gif", width=200)
st.markdown(f"## {this_year}년 {current_month}월 비용 분석")
t1, t2, t3, t4, t5, t6, t7 = st.tabs(['사용량 원단위 추이_포항', '사용량 원단위 추이_충주1','사용량 원단위 추이_충주2','단가 추이', '월 평균 클레임 지급액','당월 클레임 내역','영업외 비용 내역'])
st.divider()
#------------------------------------------------------------------------------------------------

# =========================
# 사용량 원단위 추이_포항
# =========================


with t1:
    st.markdown("### 1) 부재료 사용량 원단위 (포항)", unsafe_allow_html=True)

    file_name = st.secrets["sheets"]["f_43"]

    @st.cache_data(ttl=600)
    def load_submat_df(path: str) -> pd.DataFrame:
        # 쉼표 천단위 잘 읽히도록 thousands="," 적용
        return pd.read_csv(path, encoding="utf-8", thousands=",")

    df_src = load_submat_df(file_name)

    # ===== 표 생성 (2월 ~ 데이터의 마지막 월) =====
    df_table = modules.create_material_usage_table_pohang(
        year=this_year,
        month=current_month,
        data=df_src,
        start_month=2,
        round_digits=1,   # 소수 첫째자리 반올림
    )

    # ➜ 인덱스를 컬럼으로 승격
    df_show = df_table.reset_index()
    df_show.columns.name = None

    numeric_cols = df_show.select_dtypes(include="number").columns

    styled = (
        df_show.style
        .format({col: "{:.1f}" for col in numeric_cols}, na_rep="-")
        .hide(axis="index")

        # 1) 첫 번째 열(포항) 바디 셀 강조
        .set_properties(
            subset=[df_show.columns[0]],
            **{
                "text-align": "left",
                "font-weight": "600",
                "background-color": "#f0f0f0"  # 연한 하늘색 (원하는 색으로 변경)
                # "border-left": "3px solid #3B82F6"  # 왼쪽 굵은 포인트 라인
            }
        )

        # 2) 헤더 셀(포항)도 동일 계열로 강조
        .set_table_styles([
            {
                "selector": "th.col_heading.level0.col0",  # 첫 번째 컬럼 헤더
                "props": [
                    ("background-color", "#f0f0f0"),
                    ("font-weight", "700"),
                    ("text-align", "center")
                ],
            },
            {
                "selector": "th.col_heading",  # 다른 헤더는 가운데 정렬 유지
                "props": [("text-align", "center")],
            },
        ])

        # 3) 숫자 컬럼은 우측 정렬
        .set_properties(subset=[c for c in df_show.columns if c in numeric_cols], **{"text-align": "center"})
    )



    

    st.markdown(
        f"<div style='display:flex; justify-content:center'>{styled.to_html(index=False)}</div>",
        unsafe_allow_html=True
    )

    st.divider()
    unit = "<div style='text-align:right; font-size:15px; color:#000000;'>※ 사용량원단위 : 부재료사용량/공정처리량</div>"
    st.markdown(unit, unsafe_allow_html=True)


    

    df_plot = df_table.copy()  # index: 항목들, columns: '2월'...'N월'
    months = list(df_plot.columns)
    x = np.arange(len(months))

    fig = plt.figure(figsize=(12, 4))

    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 8.2], wspace=0.02)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax     = fig.add_subplot(gs[0, 1])

    # ----- 본 그래프 -----
    lines = []
    labels = []
    for item_name, row in df_plot.iterrows():
        y = row.values.astype(float)
        ln, = ax.plot(x, y, marker='o', linewidth=2.0, markersize=6, label=item_name)
        lines.append(ln); labels.append(item_name)
        # 점 라벨(소수 1자리)
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.1f}", ha='center', va='bottom', fontsize=9, clip_on=True)

        nudge_texts_to_avoid_overlap(ax, min_sep_px=10)   # 10~14px 사이로 취향껏

    ax.set_xticks(x); ax.set_xticklabels(months, fontsize=11)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    for spine in ['left','right','top']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', alpha=0.18)          
    ax.margins(x=0.02, y=0.15)              


    ax.set_title("[포항]", fontsize=10, fontweight='bold', pad=10)


    # ----- 왼쪽 ‘표 형태’ 범례 컬럼 -----
    # 축 요소 숨기고, 빈 좌표계에 라인 아이콘 + 텍스트를 행처럼 직접 그립니다.
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, len(labels))
    ax_leg.axis('off')

    row_h = 1.0  # 행 간격
    y0 = len(labels) - 0.5
    for i, (ln, lab) in enumerate(zip(lines, labels)):
        y_pos = y0 - i*row_h
        # 색상/마커는 그래프 라인과 동일하게
        ax_leg.plot([0.02, 0.12], [y_pos, y_pos],
                    color=ln.get_color(), linewidth=3, solid_capstyle='round')
        ax_leg.plot([0.07], [y_pos], marker='o', markersize=6,
                    color=ln.get_color())
        ax_leg.text(0.16, y_pos, lab, va='center', ha='left', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)






# =========================
#사용량 원단위 추이_충주1
# =========================

with t2:
    st.markdown("### 2) 부재료 사용량 원단위 (충주1)", unsafe_allow_html=True)

    file_name = st.secrets["sheets"]["f_43"]

    @st.cache_data(ttl=600)
    def load_submat_df(path: str) -> pd.DataFrame:
        # 쉼표 천단위 잘 읽히도록 thousands="," 적용
        return pd.read_csv(path, encoding="utf-8", thousands=",")

    df_src = load_submat_df(file_name)

    # ===== 표 생성 (2월 ~ 데이터의 마지막 월) =====
    df_table = modules.create_material_usage_table_chungju1(
        year=this_year,
        month=current_month,
        data=df_src,
        start_month=2,
        round_digits=1,   # 소수 첫째자리 반올림
    )

    # ➜ 인덱스를 컬럼으로 승격
    df_show = df_table.reset_index()
    df_show.columns.name = None

    numeric_cols = df_show.select_dtypes(include="number").columns

    styled = (
        df_show.style
        .format({col: "{:.1f}" for col in numeric_cols}, na_rep="-")
        .hide(axis="index")

        # 1) 첫 번째 열(포항) 바디 셀 강조
        .set_properties(
            subset=[df_show.columns[0]],
            **{
                "text-align": "left",
                "font-weight": "600",
                "background-color": "#f0f0f0"  # 연한 하늘색 (원하는 색으로 변경)
                # "border-left": "3px solid #3B82F6"  # 왼쪽 굵은 포인트 라인
            }
        )

        # 2) 헤더 셀(포항)도 동일 계열로 강조
        .set_table_styles([
            {
                "selector": "th.col_heading.level0.col0",  # 첫 번째 컬럼 헤더
                "props": [
                    ("background-color", "#f0f0f0"),
                    ("font-weight", "700"),
                    ("text-align", "center")
                ],
            },
            {
                "selector": "th.col_heading",  # 다른 헤더는 가운데 정렬 유지
                "props": [("text-align", "center")],
            },
        ])

        # 3) 숫자 컬럼은 우측 정렬
        .set_properties(subset=[c for c in df_show.columns if c in numeric_cols], **{"text-align": "center"})
    )

    
    

    st.markdown(
        f"<div style='display:flex; justify-content:center'>{styled.to_html(index=False)}</div>",
        unsafe_allow_html=True
    )

    st.divider()
    unit = "<div style='text-align:right; font-size:15px; color:#000000;'>※ 사용량원단위 : 부재료사용량/공정처리량</div>"
    st.markdown(unit, unsafe_allow_html=True)


    

    df_plot = df_table.copy()  # index: 항목들, columns: '2월'...'N월'
    months = list(df_plot.columns)
    x = np.arange(len(months))

    fig = plt.figure(figsize=(12, 4))

    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 8.2], wspace=0.02)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax     = fig.add_subplot(gs[0, 1])

    # ----- 본 그래프 -----
    lines = []
    labels = []
    for item_name, row in df_plot.iterrows():
        y = row.values.astype(float)
        ln, = ax.plot(x, y, marker='o', linewidth=2.0, markersize=6, label=item_name)
        lines.append(ln); labels.append(item_name)
        # 점 라벨(소수 1자리)
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.1f}", ha='center', va='bottom', fontsize=9, clip_on=True)

        

        nudge_texts_to_avoid_overlap(ax, min_sep_px=10)   # 10~14px 사이로 취향껏

    ax.set_xticks(x); ax.set_xticklabels(months, fontsize=11)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    for spine in ['left','right','top']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', alpha=0.18)          
    ax.margins(x=0.02, y=0.15)              


    ax.set_title("[충주1]", fontsize=10, fontweight='bold', pad=10)


    # ----- 왼쪽 ‘표 형태’ 범례 컬럼 -----
    # 축 요소 숨기고, 빈 좌표계에 라인 아이콘 + 텍스트를 행처럼 직접 그립니다.
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, len(labels))
    ax_leg.axis('off')

    row_h = 1.0  # 행 간격
    y0 = len(labels) - 0.5
    for i, (ln, lab) in enumerate(zip(lines, labels)):
        y_pos = y0 - i*row_h
        # 색상/마커는 그래프 라인과 동일하게
        ax_leg.plot([0.02, 0.12], [y_pos, y_pos],
                    color=ln.get_color(), linewidth=3, solid_capstyle='round')
        ax_leg.plot([0.07], [y_pos], marker='o', markersize=6,
                    color=ln.get_color())
        ax_leg.text(0.16, y_pos, lab, va='center', ha='left', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# =========================
#사용량 원단위 추이_충주2
# =========================
with t3:
    st.markdown("### 3) 부재료 사용량 원단위 (충주2)", unsafe_allow_html=True)

    file_name = st.secrets["sheets"]["f_43"]

    @st.cache_data(ttl=600)
    def load_submat_df(path: str) -> pd.DataFrame:
        # 쉼표 천단위 잘 읽히도록 thousands="," 적용
        return pd.read_csv(path, encoding="utf-8", thousands=",")

    df_src = load_submat_df(file_name)

    # ===== 표 생성 (2월 ~ 데이터의 마지막 월) =====
    df_table = modules.create_material_usage_table_chungju2(
        year=this_year,
        month=current_month,
        data=df_src,
        start_month=2,
        round_digits=1,   # 소수 첫째자리 반올림
    )

    # ➜ 인덱스를 컬럼으로 승격
    df_show = df_table.reset_index()
    df_show.columns.name = None

    numeric_cols = df_show.select_dtypes(include="number").columns

    styled = (
        df_show.style
        .format({col: "{:.1f}" for col in numeric_cols}, na_rep="-")
        .hide(axis="index")

        # 1) 첫 번째 열(포항) 바디 셀 강조
        .set_properties(
            subset=[df_show.columns[0]],
            **{
                "text-align": "left",
                "font-weight": "600",
                "background-color": "#f0f0f0"  # 연한 하늘색 (원하는 색으로 변경)
                # "border-left": "3px solid #3B82F6"  # 왼쪽 굵은 포인트 라인
            }
        )

        # 2) 헤더 셀(포항)도 동일 계열로 강조
        .set_table_styles([
            {
                "selector": "th.col_heading.level0.col0",  # 첫 번째 컬럼 헤더
                "props": [
                    ("background-color", "#f0f0f0"),
                    ("font-weight", "700"),
                    ("text-align", "center")
                ],
            },
            {
                "selector": "th.col_heading",  # 다른 헤더는 가운데 정렬 유지
                "props": [("text-align", "center")],
            },
        ])

        # 3) 숫자 컬럼은 우측 정렬
        .set_properties(subset=[c for c in df_show.columns if c in numeric_cols], **{"text-align": "center"})
    )

    
    

    st.markdown(
        f"<div style='display:flex; justify-content:center'>{styled.to_html(index=False)}</div>",
        unsafe_allow_html=True
    )

    st.divider()
    unit = "<div style='text-align:right; font-size:15px; color:#000000;'>※ 사용량원단위 : 부재료사용량/공정처리량</div>"
    st.markdown(unit, unsafe_allow_html=True)



    

    df_plot = df_table.copy()  
    months = list(df_plot.columns)
    x = np.arange(len(months))

    fig = plt.figure(figsize=(12, 1.8))

    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 7], wspace=0.02)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax     = fig.add_subplot(gs[0, 1])

    # ----- 본 그래프 -----
    lines = []
    labels = []
    for item_name, row in df_plot.iterrows():
        y = row.values.astype(float)
        ln, = ax.plot(x, y, marker='o', linewidth=2.0, markersize=6, label=item_name)
        lines.append(ln); labels.append(item_name)
        # 점 라벨(소수 1자리)
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.1f}", ha='center', va='bottom', fontsize=9, clip_on=True)
        
        nudge_texts_to_avoid_overlap(ax, min_sep_px=10)   

    ax.set_xticks(x); ax.set_xticklabels(months, fontsize=11)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    for spine in ['left','right','top']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', alpha=0.18)          
    ax.margins(x=0.02, y=0.15)              


    ax.set_title("[충주2]", fontsize=10, fontweight='bold', pad=10)


    # ----- 왼쪽 ‘표 형태’ 범례 컬럼 -----
    # 축 요소 숨기고, 빈 좌표계에 라인 아이콘 + 텍스트를 행처럼 직접 그립니다.
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, len(labels))
    ax_leg.axis('off')

    row_h = 1.0  # 행 간격
    y0 = len(labels) - 0.5
    for i, (ln, lab) in enumerate(zip(lines, labels)):
        y_pos = y0 - i*row_h
        # 색상/마커는 그래프 라인과 동일하게
        ax_leg.plot([0.02, 0.12], [y_pos, y_pos],
                    color=ln.get_color(), linewidth=3, solid_capstyle='round')
        ax_leg.plot([0.07], [y_pos], marker='o', markersize=6,
                    color=ln.get_color())
        ax_leg.text(0.16, y_pos, lab, va='center', ha='left', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# =========================
#단가 추이
# =========================
# with t4:
#     st.markdown("### 4) 단가 추이", unsafe_allow_html=True)

#     file_name = st.secrets["sheets"]["f_46"]

#     @st.cache_data(ttl=600)
#     def load_submat_df(path: str) -> pd.DataFrame:
#         return pd.read_csv(path, encoding="utf-8", thousands=",")

#     df_src = load_submat_df(file_name)

#     # ===== 표 생성 (2월 ~ 데이터의 마지막 월) =====
#     df_table = modules.create_material_usage_table_unit_price(
#         year=this_year,
#         month=current_month,
#         data=df_src,
#         start_month=2,
#         round_digits=1,   # 소수 첫째자리 반올림
#     )

#     # ➜ 인덱스를 컬럼으로 승격 (헤더 한 줄)
#     df_show = df_table.reset_index()
#     df_show.columns.name = None

#     numeric_cols = df_show.select_dtypes(include="number").columns
#     first_col = df_show.columns[0]

#     # ===== 표 스타일  =====
#     styled = (
#         df_show.style
#         .format({col: "{:.1f}" for col in numeric_cols}, na_rep="-")
#         .hide(axis="index")
#         # 1) 첫 번째 열 강조 
#         .set_properties(
#             subset=[first_col],
#             **{
#                 "text-align": "left",
#                 "font-weight": "600",
#                 "background-color": "#f0f0f0",
#                 "white-space": "nowrap",
#             }
#         )
#         # 2) 헤더 스타일 
#         .set_table_styles([
#             {
#                 "selector": "th.col_heading.level0.col0",
#                 "props": [
#                     ("background-color", "#f0f0f0"),
#                     ("font-weight", "700"),
#                     ("text-align", "center")
#                 ],
#             },
#             {"selector": "th.col_heading", "props": [("text-align", "center")]},
#         ])
#         # 3) 숫자 컬럼 정렬 
#         .set_properties(subset=[c for c in df_show.columns if c in numeric_cols], **{"text-align": "center"})
#     )

#     st.markdown(
#         f"<div style='display:flex; justify-content:center'>{styled.to_html(index=False)}</div>",
#         unsafe_allow_html=True
#     )

#     st.divider()

#     # ===== 그래프=====
    

#     df_plot = df_table.copy()
#     df_plot = df_plot.apply(pd.to_numeric, errors="coerce")

#     months = list(df_plot.columns)
#     x = np.arange(len(months))

#     fig = plt.figure(figsize=(12, 5))                      # 높이 슬림
#     gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 8.2], wspace=0.02)  
#     ax_leg = fig.add_subplot(gs[0, 0])
#     ax     = fig.add_subplot(gs[0, 1])

#     # ----- 본 그래프 -----
#     lines, labels = [], []
#     for item_name, row in df_plot.iterrows():
#         y = row.values.astype(float)
#         ln, = ax.plot(x, y, marker='o', linewidth=2.0, markersize=6, label=item_name)
#         lines.append(ln); labels.append(item_name)
#         # 점 라벨(소수 1자리)
#         for xi, yi in zip(x, y):
#             if np.isfinite(yi):
#                 ax.text(xi, yi, f"{yi:.1f}", ha='center', va='bottom', fontsize=9, clip_on=True)
        
        
#         nudge_texts_to_avoid_overlap(ax, min_sep_px=10)   # 10~14px 사이로 취향껏


#     ax.set_xticks(x); ax.set_xticklabels(months, fontsize=11)
#     ax.tick_params(axis='y', which='both', left=False, labelleft=False)
#     for spine in ['left','right','top']:
#         ax.spines[spine].set_visible(False)
#     ax.grid(axis='y', alpha=0.18)
#     ax.margins(x=0.02, y=0.15)
 

#     # ----- 왼쪽 ‘표 형태’ 범례 -----
#     ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, len(labels))
#     ax_leg.axis('off')
#     row_h = 1.0
#     y0 = len(labels) - 0.5
#     for i, (ln, lab) in enumerate(zip(lines, labels)):
#         y_pos = y0 - i*row_h
#         ax_leg.plot([0.02, 0.12], [y_pos, y_pos], color=ln.get_color(), linewidth=3, solid_capstyle='round')
#         ax_leg.plot([0.07], [y_pos], marker='o', markersize=6, color=ln.get_color())
#         ax_leg.text(0.16, y_pos, lab, va='center', ha='left', fontsize=11)

#     plt.tight_layout()
#     st.pyplot(fig, use_container_width=True)

with t4:
    st.markdown("### 4) 단가 추이", unsafe_allow_html=True)

    file_name = st.secrets["sheets"]["f_46"]

    @st.cache_data(ttl=600)
    def load_submat_df(path: str) -> pd.DataFrame:
        return pd.read_csv(path, encoding="utf-8", thousands=",")

    df_src = load_submat_df(file_name)

    # ===== 표 생성 (2월 ~ 데이터의 마지막 월) =====
    df_table = modules.create_material_usage_table_unit_price(
        year=this_year,
        month=current_month,
        data=df_src,
        start_month=2,
        round_digits=1,   # 소수 첫째자리 반올림
    )

    # ➜ 인덱스를 컬럼으로 승격 (헤더 한 줄)
    df_show = df_table.reset_index()
    df_show.columns.name = None

    numeric_cols = df_show.select_dtypes(include="number").columns
    first_col = df_show.columns[0]

    # ===== 표 스타일  =====
    styled = (
        df_show.style
        .format({col: "{:.1f}" for col in numeric_cols}, na_rep="-")
        .hide(axis="index")
        # 1) 첫 번째 열 강조 
        .set_properties(
            subset=[first_col],
            **{
                "text-align": "left",
                "font-weight": "600",
                "background-color": "#f0f0f0",
                "white-space": "nowrap",
            }
        )
        # 2) 헤더 스타일 
        .set_table_styles([
            {
                "selector": "th.col_heading.level0.col0",
                "props": [
                    ("background-color", "#f0f0f0"),
                    ("font-weight", "700"),
                    ("text-align", "center")
                ],
            },
            {"selector": "th.col_heading", "props": [("text-align", "center")]},
        ])
        # 3) 숫자 컬럼 정렬 
        .set_properties(subset=[c for c in df_show.columns if c in numeric_cols], **{"text-align": "center"})
    )

    st.markdown(
        f"<div style='display:flex; justify-content:center'>{styled.to_html(index=False)}</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # ===== 그래프=====
    # try:
    #     mpl.rcParams['font.family'] = 'Malgun Gothic'
    #     mpl.rcParams['axes.unicode_minus'] = False
    # except Exception:
    #     pass

    df_plot = df_table.copy()
    df_plot = df_plot.apply(pd.to_numeric, errors="coerce")

    months = list(df_plot.columns)
    x = np.arange(len(months))

    fig = plt.figure(figsize=(12, 4))

    # 기존: [1.8, 8.2]  →  왼쪽(표) 더 얇게
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 7.0], wspace=0.08)  
    ax_leg = fig.add_subplot(gs[0, 0])
    ax     = fig.add_subplot(gs[0, 1])
    ax.margins(x=0.12, y=0.15)
    


    
    
    
    


    # ----- 본 그래프 -----

    # 전체 데이터(질소/전력 포함)로 원래 메인 스케일 고정
    all_vals = df_plot.to_numpy(dtype=float)
    finite = np.isfinite(all_vals)
    orig_ymin = np.nanmin(all_vals[finite])
    orig_ymax = np.nanmax(all_vals[finite])
    pad = max(3, (orig_ymax - orig_ymin) * 0.15)  # 기존 느낌대로 살짝 여유
    # ↓ 메인 축 스케일을 '먼저' 고정
    ax.set_ylim(orig_ymin - pad, orig_ymax + pad)


    n2_label, pw_label = "질소(천㎥)", "전력(천kwh)"
    focus = {n2_label, pw_label}

    lines = []
    labels = []
    for item_name, row in df_plot.iterrows():
        if item_name in focus:
            continue  # 메인에서는 질소/전력 원 스케일 라인 미표시
        y = row.values.astype(float)
        ln, = ax.plot(x, y, marker='o', linewidth=2.0, markersize=6, label=item_name)
        lines.append(ln); labels.append(item_name)
        # 점 라벨
        for xi, yi in zip(x, y):
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.1f}", ha='center', va='bottom', fontsize=9, clip_on=True)

    # 겹침 보정과 메인 축 스타일
    nudge_texts_to_avoid_overlap(ax, min_sep_px=10)
    ax.set_xticks(x); ax.set_xticklabels(months, fontsize=11)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    for spine in ['left','right','top']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', alpha=0.18)
    ax.margins(x=0.02, y=0.15)

    # ===== 밴드(질소/전력 전용, 같은 축 내부에 시각적 스케일링) =====
    # 메인 축 최종 범위 기준으로 밴드 위치 계산
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    band_lo = ymin + yr * 0.12
    band_hi = ymin + yr * 0.24

    # 두 시리즈의 '공통' min/max로 같은 변환(실제 차이를 유지)
    def to_band_shared(y, lo, hi, ymin_all, ymax_all):
        y = np.asarray(y, dtype=float)
        m = np.isfinite(y)
        if not m.any() or not np.isfinite(ymin_all) or not np.isfinite(ymax_all) or ymax_all == ymin_all:
            out = np.full_like(y, (lo + hi) / 2.0)
            out[~m] = np.nan
            return out
        out = (y - ymin_all) / (ymax_all - ymin_all) * (hi - lo) + lo
        out[~m] = np.nan
        return out

    # 원값
    y_n2 = df_plot.loc[n2_label].values.astype(float)
    y_pw = df_plot.loc[pw_label].values.astype(float)

    # 공통 범위 계산
    both = np.r_[y_n2[np.isfinite(y_n2)], y_pw[np.isfinite(y_pw)]]
    ymin_all, ymax_all = (np.nanmin(both), np.nanmax(both)) if both.size else (np.nan, np.nan)

    # 밴드 좌표로 변환
    y_n2_band = to_band_shared(y_n2, band_lo, band_hi, ymin_all, ymax_all)
    y_pw_band = to_band_shared(y_pw, band_lo, band_hi, ymin_all, ymax_all)

    # 색상(라인만 색, 라벨은 검정)
    c_n2 = "#FFD400"  # 질소(노랑)
    c_pw = "#BB2649"  # 전력(버건디)

    # 밴드 라인 그리기
    ln_n2, = ax.plot(x, y_n2_band, marker="o", linewidth=2.4, color=c_n2, zorder=4, label=n2_label)
    ln_pw, = ax.plot(x, y_pw_band, marker="o", linewidth=2.4, color=c_pw, zorder=4, label=pw_label)

    # 수치 라벨은 '항상 검정'
    from matplotlib import patheffects as pe
    pe_white = [pe.withStroke(linewidth=2, foreground="white")]  # 가독성(선택)

    for xi, (vy, vyb) in enumerate(zip(y_n2, y_n2_band)):
        if np.isfinite(vy):
            ax.text(x[xi], vyb, f"{vy:.1f}", ha="center", va="bottom",
                    fontsize=9, color="#000000", clip_on=False, zorder=5, path_effects=pe_white)

    for xi, (vy, vyb) in enumerate(zip(y_pw, y_pw_band)):
        if np.isfinite(vy):
            ax.text(x[xi], vyb, f"{vy:.1f}", ha="center", va="top",
                    fontsize=9, color="#000000", clip_on=False, zorder=5, path_effects=pe_white)

    # 밴드 경계 가이드(선택)
    ax.hlines([band_lo, band_hi], x[0]-0.2, x[-1]+0.2, lw=1, alpha=0.15, zorder=1)

    # 범례용 핸들 추가(왼쪽 표/legend 렌더에 사용)
    lines += [ln_n2, ln_pw]
    labels += [n2_label, pw_label]

    # 끝값 잘림 방지(밴드 라벨 보호)
    ax.margins(x=0.06)









    


    # 범례 컬럼
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, len(labels))
    ax_leg.axis('off')
    
    row_h = 1.0
    y0 = len(labels) - 0.5
    for i, (ln, lab) in enumerate(zip(lines, labels)):
        y_pos = y0 - i*row_h

        ax_leg.plot([0.02, 0.12], [y_pos, y_pos],
                    color=ln.get_color(), linewidth=3, solid_capstyle='round')
        ax_leg.plot([0.07], [y_pos], marker='o', markersize=6,
                    color=ln.get_color())
        ax_leg.text(0.16, y_pos, lab, va='center', ha='left', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# =========================
#월 평균 클레임 지급액
# =========================
with t5:
    st.markdown(f"<h4>1. 월 평균 클레임 지급액</h4>", unsafe_allow_html=True)
    df = modules.update_monthly_claim_form(this_year)

    

    styled_df = (
            df.style
            .format(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and pd.notnull(x) else x)
            .set_properties(**{'text-align': 'right'})
            # .set_properties(**{'font-family': 'Noto Sans KR'})
            .set_properties(**{
                'font-family': "'Noto Sans KR','NanumGothic','Noto Sans CJK KR','Malgun Gothic','AppleGothic',sans-serif"
            })
            
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