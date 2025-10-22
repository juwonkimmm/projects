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



    df_plot = df_table.copy()
    months  = list(df_plot.columns)          # x축 라벨
    x       = list(months)

    fig = go.Figure()

    # ----- 본 그래프 -----
    for item_name in df_plot.index:
        y = pd.to_numeric(df_plot.loc[item_name], errors="coerce").values.astype(float)

        # 라벨 겹침 완화
        textpos = ["top center" if i % 2 == 0 else "bottom center" for i in range(len(y))]

        fig.add_trace(go.Scatter(
            x=x, y=y, name=item_name,
            mode="lines+markers+text",
            line=dict(width=2),
            marker=dict(size=6),
            text=[f"{v:.1f}" if np.isfinite(v) else "" for v in y],
            textposition=textpos,
            textfont=dict(size=11),
            hovertemplate="월=%{x}<br>값=%{y:.1f}<extra></extra>",
        ))

    # ----- 레이아웃 (왼쪽 표형 범례, y축 라벨 숨김, 스타일) -----
    fig.update_layout(
        title="[포항]",
        template="plotly_white",
        margin=dict(l=150, r=20, t=40, b=40),  # 왼쪽 범례 자리
        hovermode="x unified",
        font=dict(family="Noto Sans KR", size=12),

        # 범례를 왼쪽 '바깥'에 세로로
        legend=dict(
            orientation="v",
            x=-0.15, y=1.0, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=1
        )
    )

    # 축/그리드: Matplotlib 설정과 유사하게
    fig.update_xaxes(
        title_text=None,
        tickmode="array", tickvals=x, ticktext=x,
        showgrid=False  # 필요 시 True
    )
    fig.update_yaxes(
        title_text=None,
        showticklabels=False,          # y축 눈금 라벨 숨김
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)"   # y 그리드 옅게
    )

    # Streamlit
    st.plotly_chart(fig, use_container_width=True)







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




    # df_plot = df_table.copy()  # index: 항목들, columns: '2월'...'N월'
    # months = list(df_plot.columns)
    # x = np.arange(len(months))

    # fig = plt.figure(figsize=(12, 4))

    # gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 8.2], wspace=0.02)
    # ax_leg = fig.add_subplot(gs[0, 0])
    # ax     = fig.add_subplot(gs[0, 1])

    # # ----- 본 그래프 -----
    # lines = []
    # labels = []
    # for item_name, row in df_plot.iterrows():
    df_plot = df_table.copy()
    months  = list(df_plot.columns)      # x축 라벨
    x       = months                     # 카테고리형 x

    fig = go.Figure()

    # ----- 본 그래프: 라인+마커+값 라벨 -----
    for item_name in df_plot.index:
        y = pd.to_numeric(df_plot.loc[item_name], errors="coerce").values.astype(float)

        # 라벨 겹침 완화: 위/아래 교차 배치 (원하면 "top center"로 고정)
        textpos = ["top center" if i % 2 == 0 else "bottom center" for i in range(len(y))]

        fig.add_trace(go.Scatter(
            x=x, y=y, name=item_name,
            mode="lines+markers+text",
            line=dict(width=2),
            marker=dict(size=6),
            text=[f"{v:.1f}" if np.isfinite(v) else "" for v in y],
            textposition=textpos,
            textfont=dict(size=11),
            hovertemplate="월=%{x}<br>값=%{y:.1f}<extra></extra>",
        ))

    # ----- 레이아웃 / 축 -----
    fig.update_layout(
        title="[충주1]",
        template="plotly_white",
        margin=dict(l=150, r=20, t=50, b=40),   # 왼쪽 범례 자리
        hovermode="x unified",
        font=dict(family="Noto Sans KR", size=12),
        legend=dict(                            # 왼쪽 '표형' 범례
            orientation="v",
            x=-0.15, y=1.0, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=1
        )
    )

    # x축: 카테고리 라벨 그대로
    fig.update_xaxes(
        tickmode="array", tickvals=x, ticktext=x,
        showgrid=False
    )
    # y축: 라벨 숨김 + 옅은 그리드
    fig.update_yaxes(
        showticklabels=False,
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)"
    )

    # (선택) 여백 살짝: matplotlib의 ax.margins 유사 효과
    # 필요시 y범위 패딩 주기
    all_vals = pd.to_numeric(df_plot.values.ravel(), errors="coerce").astype(float)
    finite = np.isfinite(all_vals)
    if finite.any():
        ymin = float(np.nanmin(all_vals[finite])); ymax = float(np.nanmax(all_vals[finite]))
        pad = max((ymax - ymin) * 0.15, 1e-9)     # matplotlib의 y=0.15 마진 유사
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    # Streamlit 출력
    st.plotly_chart(fig, use_container_width=True)

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



    try:
        mpl.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글
        mpl.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    # df_plot = df_table.copy()  
    # months = list(df_plot.columns)
    # x = np.arange(len(months))

    # fig = plt.figure(figsize=(12, 1.8))

    # gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 7], wspace=0.02)
    # ax_leg = fig.add_subplot(gs[0, 0])
    # ax     = fig.add_subplot(gs[0, 1])

    # # ----- 본 그래프 -----
    # lines = []
    # labels = []
    # for item_name, row in df_plot.iterrows():

    df_plot = df_table.copy()                     # index: 항목, columns: '2월'...'N월'
    months  = list(df_plot.columns)               # x축 카테고리
    x       = months

    fig = go.Figure()

    # ----- 본 그래프: 라인 + 마커 + 값 라벨 -----
    for i, item_name in enumerate(df_plot.index):
        y = pd.to_numeric(df_plot.loc[item_name], errors="coerce").values.astype(float)

        # 라벨 겹침 완화: 위/아래 교차 (필요시 "top center"로 고정)
        textpos = ["top center" if j % 2 == 0 else "bottom center" for j in range(len(y))]

        fig.add_trace(go.Scatter(
            x=x, y=y, name=item_name,
            mode="lines+markers+text",
            line=dict(width=2),
            marker=dict(size=6),
            text=[f"{v:.1f}" if np.isfinite(v) else "" for v in y],
            textposition=textpos,
            textfont=dict(size=11),
            hovertemplate="월=%{x}<br>값=%{y:.1f}<extra></extra>",
        ))

    # ----- 레이아웃/축 -----
    fig.update_layout(
        title="[충주2]",
        template="plotly_white",
        height=260,                                 # figsize(12, 1.8) 느낌으로 낮게
        margin=dict(l=150, r=20, t=50, b=30),       # 왼쪽 범례 자리
        hovermode="x unified",
        uniformtext_minsize=9, uniformtext_mode="hide",
        font=dict(family="Noto Sans KR", size=12),

        # 왼쪽 ‘표형’ 범례 (세로, 바깥)
        legend=dict(
            orientation="v",
            x=-0.15, y=1.0, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=1
        )
    )

    # x축: 카테고리 틱
    fig.update_xaxes(
        tickmode="array", tickvals=x, ticktext=x,
        showgrid=False
    )

    # y축: 눈금 라벨 숨김 + 옅은 그리드 + 마진 유사 패딩
    fig.update_yaxes(
        showticklabels=False,
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.18)"
    )

    all_vals = pd.to_numeric(df_plot.values.ravel(), errors="coerce").astype(float)
    finite = np.isfinite(all_vals)
    if finite.any():
        ymin = float(np.nanmin(all_vals[finite])); ymax = float(np.nanmax(all_vals[finite]))
        pad = max((ymax - ymin) * 0.15, 1e-9)
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    # Streamlit 출력
    st.plotly_chart(fig, use_container_width=True)

# =========================
#단가 추이
# =========================
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

    # ----- 그래프 불러오기 -----

    df_plot = df_table.copy()
    df_plot = df_plot.apply(pd.to_numeric, errors="coerce")

    


    # ----- 본 그래프 -----
    from plotly.subplots import make_subplots

    n2_label, pw_label = "질소(천㎥)", "전력(천kwh)"
    x = list(months)
    others = [idx for idx in df_plot.index if idx not in {n2_label, pw_label}]

    # --- 데이터 ---
    y_n2 = pd.to_numeric(df_plot.loc[n2_label], errors="coerce").values.astype(float)
    y_pw = pd.to_numeric(df_plot.loc[pw_label], errors="coerce").values.astype(float)
    others_vals = (pd.to_numeric(df_plot.loc[others].values.ravel(), errors="coerce")
                .astype(float) if len(others) else np.array([], dtype=float))

    def rng(v, pad_ratio=0.06):
        f = np.isfinite(v)
        if not f.any(): return [0.0, 1.0]
        vmin, vmax = float(np.nanmin(v[f])), float(np.nanmax(v[f]))
        pad = max((vmax - vmin) * pad_ratio, 1e-9)
        return [vmin - pad, vmax + pad]

    y1_range = rng(others_vals)  # 질소/전력 제외
    y2_range = rng(np.r_[y_n2[np.isfinite(y_n2)], y_pw[np.isfinite(y_pw)]])  # 질소/전력용

    # --- 2행 1열 서브플롯: x축 공유 ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,          
        row_heights=[0.80, 0.25]       
    )

    # 질소 전력 제외
    base_colors = ["#1f77b4","#ff7f0e","#939393","#bcbd22","#000080","#009900"]
    for i, name in enumerate(others):
        y = pd.to_numeric(df_plot.loc[name], errors="coerce").values.astype(float)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, name=name,
                mode="lines+markers+text",                 
                line=dict(width=2, color=base_colors[i % len(base_colors)]),
                marker=dict(size=6),
                texttemplate="%{y:.1f}",                   
                textposition="top center",                 
                textfont=dict(size=10),
                hovertemplate="월=%{x}<br>값=%{y:.1f}<extra></extra>",
            ),
            row=1, col=1
        )

    # (2) 아래: 질소/전력 (원값 그대로)
    fig.add_trace(
        go.Scatter(
            x=x, y=y_n2, name=n2_label,
            mode="lines+markers+text",                    
            line=dict(width=2.4, color="#FFD400"), marker=dict(size=7),
            texttemplate="%{y:.1f}",
            textposition="top center",                 
            textfont=dict(size=10),
            hovertemplate="월=%{x}<br>값=%{y:.1f}<extra></extra>",
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_pw, name=pw_label,
            mode="lines+markers+text",                    # ← text 추가
            line=dict(width=2.4, color="#BB2649"), marker=dict(size=7),
            texttemplate="%{y:.1f}",
            textposition="top center",                 # ← 전력은 아래(겹침 완화)
            textfont=dict(size=10),
            hovertemplate="월=%{x}<br>값=%{y:.1f}<extra></extra>",
        ),
        row=2, col=1
    )

    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")


    # --- 레이아웃/축/범례 ---
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=20, t=60, b=40),
        font=dict(family="Noto Sans KR", size=40),
        hovermode="x unified",
        legend=dict(
            orientation="v", x=-0.15, y=1.0, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.7)"
        )
    )
    fig.update_yaxes(domain=[0.00, 0.25], row=2, col=1)   
    fig.update_yaxes(domain=[0.24, 1.00], row=1, col=1) 


    # (B) 아래 패널 내에서 하단 정렬 느낌
    r_vals = np.r_[y_n2[np.isfinite(y_n2)], y_pw[np.isfinite(y_pw)]]
    rmin, rmax = float(np.nanmin(r_vals)), float(np.nanmax(r_vals))
    span = max(rmax - rmin, 1e-9)
    fig.update_yaxes(range=[rmin - span*0.05, rmax + span*1.20], row=2, col=1)



    st.plotly_chart(fig, use_container_width=True)

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
    st.markdown("<h4>7) 영업외 비용 (최근 3개월)</h4>", unsafe_allow_html=True)

    # 데이터 로드 (secrets)
    csv_src = st.secrets['sheets']['f_49']
    df_raw = modules.load_nonop_cost_csv(csv_src)

    # 표 생성: 구분2 섹션(기타비용/금융비용) × 구분4 세부항목
    df_tbl = modules.create_nonop_cost_3month_by_g2_g4(
        year=this_year,
        month=current_month,
        data=df_raw
    )

    # 스타일
    num_cols = [c for c in df_tbl.columns if c not in ("구분","계정","_row_type")]

    def _fmt(x):
        if isinstance(x, (int, float)):
            return f"{x:,.0f}"
        return x

    sty = (
        df_tbl.drop(columns=[c for c in ["_row_type"] if c in df_tbl.columns])
             .style
             .format(_fmt, subset=pd.IndexSlice[:, num_cols])
             .set_properties(**{"text-align":"right", "font-family":"Noto Sans KR"})
             .set_properties(subset=pd.IndexSlice[:, ["구분","계정"]], **{"text-align":"left"})
    )

    right_idx = df_tbl.index[
    df_tbl["계정"].astype(str).str.strip().isin(["고철매각작업비","기타"])
    ]
    sty = sty.set_properties(
        subset=(right_idx, ["계정"]),
        **{"text-align":"right", "padding-right":"12px"}
    )

    

    # 행 하이라이트: 섹션 합계/총계 굵게+회색, 세부항목은 기본
    def _row_style(row):
        t = df_tbl.loc[row.name, "_row_type"]
        if t == "section_total":
            return ["font-weight:600; background-color:#f5f5f5"] * len(row)
        if t == "grand_total":
            return ["font-weight:700; background-color:#ededed"] * len(row)
        return [""] * len(row)
    
    sty = sty.apply(_row_style, axis=1)

    styles_prod = []

    # 잡손실 기타 고철매각작업비 행 작업
    styles_prod.append({
            'selector': 'tbody tr:nth-child(6) td:nth-child(2)',
            'props': [('border-bottom', '2px solid white !important')]
            
    })

    styles_prod.append({
            'selector': 'tbody tr:nth-child(7) td:nth-child(2)',
            'props': [('border-bottom', '2px solid white !important')]
            
    })

    #기타비용 위 빈 행들 작업
    for i in range(1, 11):
        styles_prod.append({
        "selector": f"tbody tr:nth-child({i}) td:nth-of-type(1)",
        "props": [("border-bottom", "2px solid white !important")]
    })
        
    #금융비용 위 빈 행들 작업
    for i in range(12, 19):
        styles_prod.append({
        "selector": f"tbody tr:nth-child({i}) td:nth-of-type(1)",
        "props": [("border-bottom", "2px solid white !important")]
    })
        
    
    styles_prod.append({
            'selector': 'tbody tr:nth-child(11)',
            'props': [('border-bottom', '3px solid gray !important')]
            
    })

    for i in range(1, 12):
        styles_prod.append({
        "selector": f"tbody tr:nth-child({i})",
        "props": [("border-right", "3px solid gray !important")]
    })
        
    for i in range(12, 21):
        styles_prod.append({
        "selector": f"tbody tr:nth-child({i})",
        "props": [("border-right", "3px solid gray !important")]
    })
        
    styles_prod.append({
            'selector': 'tbody tr:nth-child(20)',
            'props': [('border-bottom', '3px solid gray !important')]
            
    })
    

    sty = sty.set_table_styles(styles_prod, overwrite=False)

     
    # 음수는 빨간색
    def _neg_red(v):
        try:
            return "color:#0000FF" if float(v) < 0 else ""
        except Exception:
            return ""
    for c in num_cols:
        sty = sty.applymap(_neg_red, subset=pd.IndexSlice[:, [c]])


    if hasattr(sty, "hide"):          
        sty = sty.hide(axis="index")  # 인덱스와 인덱스 헤더 셀 모두 제거


    table_html = sty.to_html(index=False)
    st.markdown(f"<div style='display:flex; justify-content:center;'>{table_html}</div>", unsafe_allow_html=True)
    

# Footer
st.markdown("""
<style>.footer { bottom: 0; left: 0; right: 0; padding: 8px; text-align: center; font-size: 13px; color: #666666;}</style>
<div class="footer">ⓒ 2025 SeAH Special Steel Corp. All rights reserved.</div>
""", unsafe_allow_html=True)