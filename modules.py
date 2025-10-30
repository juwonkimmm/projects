import re
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
# from typing import List, Tuple  # (파이썬 3.8/3.9 호환용 필요 시 주석 해제)

this_year = datetime.today().year
current_month = datetime.today().month

# ---------------------------------------------
# 공통 유틸 (사이드바/인덱스)
# ---------------------------------------------
def date_update_callback():
    st.session_state.year = st.session_state.year_selector
    st.session_state.month = st.session_state.month_selector

def create_sidebar():
    with st.sidebar:
        st.title("날짜 선택")
        if 'year' not in st.session_state:
            st.session_state.year = this_year
        if 'month' not in st.session_state:
            st.session_state.month = current_month

        st.selectbox(
            '년(Year)', range(2020, 2031),
            key='year_selector',
            index=st.session_state.year - 2020,
            on_change=date_update_callback
        )
        st.selectbox(
            '월(Month)', range(1, 13),
            key='month_selector',
            index=st.session_state.month - 1,
            on_change=date_update_callback
        )
        st.info(f"선택된 날짜: {st.session_state.year}년 {st.session_state.month}월")

def get_month_index(year, month):
    """
    year, month 기준으로 '최근 12개월'의 말일 인덱스를 안전하게 생성.
    12월 → 'YYYY-13' 문제 방지.
    """
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    date_index = pd.date_range(end=end, periods=12, freq='M')
    return [f"{d.year % 100}년 {d.month}월" for d in date_index]

def get_year_mean_index(year):
    end_date = f"{year}-11"
    date_index = pd.date_range(end=end_date, periods=5, freq='Y')
    return [f"{date.year % 100}년 월평균" for date in date_index]

def get_year_end_index(year):
    end_date = f"{year}-11"
    date_index = pd.date_range(end=end_date, periods=5, freq='Y')
    return [f"{date.year % 100}년말" for date in date_index]

def create_df(year, month, data, mean="True", prev_year=2, prev_month=4):
    months = get_month_index(year, month)
    years = get_year_mean_index(year) if mean == "True" else get_year_end_index(year)
    columns = years[-prev_year:] + months[-prev_month:]

    categories = [col for col in data.columns if '구분' in col][-2:]
    index = data[categories].drop_duplicates()
    indexes = pd.MultiIndex.from_frame(index, names=['', ''])

    df = pd.DataFrame(0, index=indexes, columns=columns)

    for i in df.columns:
        if '월평균' in i:
            df.loc[:, i] = data[(data['연도'] == int(f'{20}{i[:2]}')) & (data['월'] == '월평균')]['실적'].values
        elif '년말' in i:
            df.loc[:, i] = data[(data['연도'] == int(f'{20}{i[:2]}')) & (data['월'] == '12월')]['실적'].values
        else:
            df.loc[:, i] = data[(data['연도'] == int(f'{20}{i[:2]}')) & (data['월'] == i.split(" ")[1])]['실적'].values
    return df
# --- 공통: 컬럼 세트 생성 유틸 ---
def _build_defect_cols(year:int, month:int) -> list[str]:
    prev = f"{str(year-1)[-2:]}년 월평균"
    target = f"{str(year)[-2:]}년 목표"
    months = [f"{m}월" for m in range(1, month+1)]
    return [prev, target] + months + ["합계", "월평균"]


# ---------------------------------------------
# 매출 집계표
# ---------------------------------------------
def create_report_form(year):
    outside_index = ['CHQ', 'CD', 'STS', 'BTB', 'PB', '기타', '합계']
    inside_index = ['금액', '중량']
    hier_index = pd.MultiIndex.from_product([outside_index, inside_index])
    hier_column = pd.MultiIndex.from_tuples([
        (' ', f"'{str(year)[-2:]}년 계획"), (' ', '전월'),
        ('당월', '계획'), ('당월', '실적'), ('당월', '계획대비'), ('당월', '전월대비'),
        ('당월누적', '계획'), ('당월누적', '실적'), ('당월누적', '계획대비')
    ])
    df = pd.DataFrame(0, index=hier_index, columns=hier_column)
    df = df.drop(index=[('기타', '중량')])
    return df

def update_report_form(year, month):
    df = create_report_form(year)
    file_name = st.secrets["sheets"]["f_30"]
    agg = pd.read_csv(file_name, thousands=',')
    agg.loc[agg['구분2'] == '금액', '실적'] /= 1_000_000
    agg.loc[agg['구분2'] == '중량', '실적'] /= 1_000
    agg['실적'] = round(agg['실적']).astype(float)

    end_date = f"{year}-{month}"
    date_index = pd.date_range(end=end_date, periods=6, freq='M')
    year_month_index = [f"{d.year} {d.month}" for d in date_index]

    order = ['CHQ', 'CD', 'STS', 'BTB', 'PB', '기타']

    temp = agg[(agg['연도'] == year) & (agg['구분3'] == '계획')]
    df.iloc[:-2, 0] = temp.groupby(['구분1', '구분2'])['실적'].sum().reindex(order, level=0)

    temp = agg[(agg['연도'] == int(year_month_index[-1].split(" ")[0])) &
               (agg['월'] == int(year_month_index[-1].split(" ")[1])) &
               (agg['구분3'] == '실적')]
    df.iloc[:-2, 1] = temp.groupby(['구분1', '구분2'])['실적'].sum().reindex(order, level=0)

    temp = agg[(agg['연도'] == year) & (agg['월'] == month) & (agg['구분3'] == '계획')]
    df.iloc[:-2, 2] = temp.groupby(['구분1', '구분2'])['실적'].sum().reindex(order, level=0)

    temp = agg[(agg['연도'] == year) & (agg['월'] == month) & (agg['구분3'] == '실적')]
    df.iloc[:-2, 3] = temp.groupby(['구분1', '구분2'])['실적'].sum().reindex(order, level=0)

    temp = agg[(agg['연도'] == year) & (agg['월'] <= month) & (agg['구분3'] == '계획')]
    df.iloc[:-2, 6] = temp.groupby(['구분1', '구분2'])['실적'].sum().reindex(order, level=0)

    temp = agg[(agg['연도'] == year) & (agg['월'] <= month) & (agg['구분3'] == '실적')]
    df.iloc[:-2, 7] = temp.groupby(['구분1', '구분2'])['실적'].sum().reindex(order, level=0)

    df.iloc[-2, :] = df.iloc[[0, 2, 4, 6, 8, 10]].sum()
    df.iloc[-1, :] = df.iloc[[1, 3, 5, 7, 9]].sum()

    row_1 = df.iloc[[0, 2, 4, 6, 8, 11]]
    row_2 = df.iloc[[1, 3, 5, 7, 9, 12]]
    new_df = round(((row_1.reset_index(drop=True) / row_2.reset_index(drop=True)) * 1000), 0)
    new_df.index = pd.MultiIndex.from_tuples(
        [('CHQ', '단가'), ('CD', '단가'), ('STS', '단가'), ('BTB', '단가'), ('PB', '단가'), ('합계', '단가')]
    )
    df = pd.concat([df, new_df])

    level1_order = ['CHQ', 'CD', 'STS', 'BTB', 'PB', '기타', '합계']
    level2_order = ['금액', '단가', '중량']
    df.index = pd.MultiIndex.from_arrays([
        pd.Categorical(df.index.get_level_values(0), categories=level1_order, ordered=True),
        pd.Categorical(df.index.get_level_values(1), categories=level2_order, ordered=True)
    ])
    df = df.sort_index()

    df.iloc[:, 4] = round(((df.iloc[:, 3] / df.iloc[:, 2]) - 1) * 100, 1)
    df.iloc[:, 4] = df.iloc[:, 4].apply(lambda x: f"{x}%")
    df.iloc[:, 5] = round(((df.iloc[:, 3] / df.iloc[:, 1]) - 1) * 100, 1)
    df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: f"{x}%")
    df.iloc[:, 8] = round(((df.iloc[:, 7] / df.iloc[:, 6]) - 1) * 100, 1)
    df.iloc[:, 8] = df.iloc[:, 8].apply(lambda x: f"{x}%")

    return df.fillna(0)

# ---------------------------------------------
# 등급별 판매구성
# ---------------------------------------------
def update_item_form(df):
    df.loc[('정상입고품', ''), :] = df.iloc[0, :] + df.iloc[1, :]
    df.loc[('정상입고품', '구성비'), :] = round(
        (df.loc[('정상입고품', '산업재 혹은 중국재'), :] / df.loc[('정상입고품', ''), :]) * 100, 1
    )
    df.loc[('합계', ''), :] = df.iloc[[0, 1, 2]].sum()
    df.loc[('B급', '구성비'), :] = round(
        (df.loc[('B급', ''), :] / df.loc[('합계', ''), :]) * 100, 1
    )

    df = df.rename(index={'정상입고품': '정품', '산업재 혹은 중국재': '산업/중국재'})
    sort_order = [
        ('정품', '정상'), ('정품', '산업/중국재'), ('정품', '구성비'), ('정품', ''),
        ('B급', ''), ('B급', '구성비'), ('합계', '')
    ]
    df = df.reindex(sort_order)

    df.loc[:, ('전월대비')] = df.iloc[:, 5] - df.iloc[:, 4]
    df.loc[:, ('%')] = (df.iloc[:, 6] / df.iloc[:, 4]) * 100
    df.loc[:, ('%')] = round(df.loc[:, ('%')].astype(float), 1).apply(lambda x: f"{x}%")
    df.loc[('정품', '구성비'), ('전월대비')] = round(df.loc[('정품', '구성비'), ('전월대비')], 2)
    df.loc[('B급', '구성비'), ('전월대비')] = round(df.loc[('B급', '구성비'), ('전월대비')], 2)
    df.loc[('정품', '구성비'), df.columns[:-1]] = df.loc[('정품', '구성비'), df.columns[:-1]].apply(lambda x: f"{x}%")
    df.loc[('B급', '구성비'), df.columns[:-1]] = df.loc[('B급', '구성비'), df.columns[:-1]].apply(lambda x: f"{x}%")
    df.loc[('정품', '구성비'), '%'] = ' '
    df.loc[('B급', '구성비'), '%'] = ' '
    df.index.names = [None, None]
    return df

# ---------------------------------------------
# PSI
# ---------------------------------------------
def create_psi_form(year, month):
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    date_index = pd.date_range(end=end, periods=12, freq='M')
    index = [f"{d.year % 100}.{d.month}" for d in date_index]
    columns = ['원재료 입고①', '매출②', '총재고③', '출고율(②/①)', '재고율(③/②)']
    return pd.DataFrame(0, index=index, columns=columns)

def update_psi_form(year, month, data):
    df = create_psi_form(year, month)
    for i in df.index:
        yy = int('20' + i.split('.')[0])
        mm = i.split('.')[1]
        vals = data[(data['연도'] == yy) & (data['월'] == f'{mm}월')]['실적'].to_list()
        ship = f"{round((vals[1] / vals[0]) * 100, 1)}%" if vals[0] != 0 else "0%"
        invt = f"{round((vals[2] / vals[1]) * 100, 1)}%" if vals[1] != 0 else "-"
        df.loc[i] = vals + [ship, invt]
    return df

def update_psi_2_form(year, month, data):
    df = create_psi_form(year, month)
    for i in df.index:
        yy = int('20' + i.split('.')[0])
        mm = i.split('.')[1]
        vals = data[(data['연도'] == yy) & (data['월'] == f'{mm}월')]['실적'].to_list()
        ship = f"{round((vals[1] / vals[0]) * 100, 1)}%" if vals[0] != 0 else "0%"
        invt = f"{round((vals[2] / vals[1]) * 100, 1)}%" if vals[1] != 0 else "-"
        df.loc[i] = vals + [ship, invt]
    return df

# ---------------------------------------------
# 재고자산 회전율
# ---------------------------------------------
def create_turnover_form(year, month):
    index_tuples = [
        ('', '상품'), ('', '제품'), ('', '(중량)'),
        (' ', '재공품'), (' ', '(중량)'),
        ('  ', '원재료'), ('  ', '(중량)'),
        ('   ', '부재료'), ('   ', '미착재료'),
        ('재고자산계', ' '), ('(중량 계)', ' ')
    ]
    hier_index = pd.MultiIndex.from_tuples(index_tuples)

    months = get_month_index(year, month)
    y3 = "20" + months[-3].split(" ")[0]; m3 = months[-3].split(" ")[1]
    y2 = "20" + months[-2].split(" ")[0]; m2 = months[-2].split(" ")[1]
    y1 = "20" + months[-1].split(" ")[0]; m1 = months[-1].split(" ")[1]

    columns = pd.MultiIndex.from_tuples([
        (' ', f'{str(year - 2)[2:]}년말'),
        (' ', f'{str(year - 1)[2:]}년말'),
        (y3 + '년', m3), (y2 + '년', m2), (y1 + '년', m1),
        ('전월대비', '증감'), ('전월대비', '증감률')
    ])
    return pd.DataFrame(0, index=hier_index, columns=columns)

def update_turnover_form(year, month):
    file_name = st.secrets['sheets']['f_50']
    turnover = pd.read_csv(file_name, thousands=',')
    turnover['실적'] = round(turnover['실적']).astype(float)
    df = create_turnover_form(year, month)

    for i in df.columns[:-2]:
        if '년말' in i[1]:
            temp = turnover[(turnover['연도'] == int('20' + i[1][:2])) & (turnover['월'] == 12)]
            df.iloc[:-2, df.columns.get_loc(i)] = temp['실적'].values
        else:
            yy = int(i[0].replace("년", ""))
            mm = int(i[1].replace("월", ""))
            temp = turnover[(turnover['연도'] == yy) & (turnover['월'] == mm)]
            df[i][:-2] = temp['실적'].values

    for r in [1, 3, 5, 7, 8]:
        df.iloc[r, :] = round(df.iloc[r, :] / 1_000_000, 0)
        df.iloc[9, :] = df.iloc[9, :] + df.iloc[r, :]
    for r in [2, 4, 6]:
        df.iloc[r, :] = round(df.iloc[r, :] / 1_000, 0)
        df.iloc[10, :] = df.iloc[10, :] + df.iloc[r, :]

    df.loc[:, ('전월대비', '증감')] = (df.iloc[:, -3] - df.iloc[:, -4]).values
    df[('전월대비', '증감률')] = round((df.iloc[:, -2] / df.iloc[:, -4]) * 100, 1)
    df = df.fillna(0)
    df.iloc[:, -1] = df.iloc[:, -1].astype(object).apply(lambda x: f"{x}%")
    return df

# ---------------------------------------------
# 별첨 실적요약
# ---------------------------------------------
def create_performance_form(year, month):
    index = ['매출액', '판매량', '영업이익', '경상이익']
    months = get_month_index(year, month)
    return pd.DataFrame(index=index, columns=months[-5:])

def update_performance_form(year, month):
    all_dfs = {}
    file_name = st.secrets['sheets']['f_89']
    data = pd.read_csv(file_name)

    data['실적'] = data['실적'].str.replace(r'\(', '-', regex=True).str.replace(r'\)', '', regex=True)
    data['실적'] = data['실적'].str.replace(",", "").astype(int)
    data.loc[data['구분2'] != '판매량', '실적'] /= 1_000_000
    data.loc[data['구분2'] == '판매량', '실적'] /= 1_000

    for category in data['구분1'].unique():
        df = create_performance_form(year, month)
        for col in df.columns:
            yy = int("20" + col.split(" ")[0].replace("년", ""))
            mm = int(col.split(" ")[1].replace("월", ""))
            temp = data[(data['연도'] == yy) & (data['월'] == mm) & (data['구분1'] == category)]
            df.loc[:, col] = round(temp['실적'], 0).astype(int).to_list()

        sales = df.loc['매출액', :].replace(0, np.nan)
        df.loc['영업이익률', :] = (df.loc['영업이익', :] / sales) * 100
        df.loc['영업이익률', :] = df.loc['영업이익률', :].astype(float).round(1).astype(str) + "%"
        all_dfs[category] = df
    return all_dfs

def update_monthly_claim_form(year):
    file_name = st.secrets['sheets']['f_47']
    claim = pd.read_csv(file_name, thousands=',')
    claim['실적'] /= 1_000_000
    df = pd.DataFrame(
        0, index=claim['구분2'].unique(),
        columns=[f"{str(y)[2:]}년" for y in claim['연도'].unique()[-5:-1]]
    )
    for i in df.index:
        result = claim.groupby(['구분2', '연도'])['실적'].mean()[i]
        for j in df.columns:
            yy = int("20" + j.replace("년", ""))
            df.loc[i, j] = round(result[yy], 0)
    df.loc['합계', :] = df.iloc[[0, 1, 2, 3, 4]].sum()
    return df

# =========================================================
# 생산: 전체 생산실적 (정렬 포함)
# =========================================================
def sort_board_like_item(df: pd.DataFrame) -> pd.DataFrame:
    """
    '등급별 판매현황'처럼 명확한 표시 순서를 강제.
    (그룹행과 공장행을 원하는 순서로 보여주기)
    """
    order = [
        ("", "포항"),
        ("", "충주"),
        ("CHQ", ""),
        ("", "포항"),
        ("", "충주2"),
        ("CD", ""),
        ("", "포항"),
        ("", "충주2"),
        ("STS", ""),
        ("BTB", ""),
        ("PB", ""),
        ("합계", ""),
        ("포항", ""),
        ("충주", ""),
        ("충주2", ""),
    ]
    names = list(df.index.names)
    target_idx = pd.MultiIndex.from_tuples(order, names=names)
    return df.reindex(target_idx)

def create_board_summary_table(year: int,
                               month: int,
                               data: pd.DataFrame,
                               base_year: int | None = None,
                               prev_year_for_avg: int | None = None) -> pd.DataFrame:
    if base_year is None:
        base_year = year
    if prev_year_for_avg is None:
        prev_year_for_avg = year - 1

    df = data.copy()

    # --- 기본 전처리 ---
    for c in ['구분1', '구분2', '구분3']:
        if c not in df.columns:
            df[c] = ''
        df[c] = df[c].fillna('').astype(str)

    # 실적 (톤 그대로)
    df['실적'] = pd.to_numeric(df.get('실적', 0), errors='coerce').fillna(0.0)

    # 연/월 캐스팅
    y_raw = df.get('연도', np.nan).astype(str).str.extract(r'(\d{4}|\d{2})')[0]
    def _to_year_int(v):
        if pd.isna(v): return np.nan
        s = str(v);   return 2000 + int(s) if len(s) == 2 else int(s)
    df['연도'] = y_raw.apply(_to_year_int)

    m_raw = (df.get('월', '')
               .astype(str).str.replace('월', '', regex=False)
               .str.replace('.', '', regex=False).str.strip())
    df['월'] = pd.to_numeric(m_raw, errors='coerce')

    df = df[df['연도'].between(1990, 2100) & df['월'].between(1, 12)]
    df['연도'] = df['연도'].astype(int)
    df['월'] = df['월'].astype(int)

    # 라벨 정규화
    def clean_text(s: str) -> str:
        s = str(s).replace('\xa0', ' ')
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    df['구분1'] = df['구분1'].map(clean_text)
    df['구분2'] = df['구분2'].map(clean_text)

    def norm_group(x: str) -> str:
        s = x.lower()
        if 'chq' in s: return 'CHQ'
        if re.search(r'(^|\s)cd(\s|$)', s): return 'CD'
        if 'sts' in s: return 'STS'
        if 'btb' in s: return 'BTB'
        if re.search(r'(^|\s)pb(\s|$)', s): return 'PB'
        return x.strip().upper()

    def norm_plant(x: str) -> str:
        s = x.replace(' ', '').replace('공장', '')
        s = re.sub(r'[\(\[\{].*?[\)\]\}]', '', s).strip()
        if '포항' in s: return '포항'
        if ('충주' in s) and ('2' in s): return '충주2'
        if '충주2' in s: return '충주2'
        if '충주' in s: return '충주'
        return x.strip()

    df['G1'] = df['구분1'].map(norm_group)
    df['G2'] = df['구분2'].map(norm_plant)

    # 집계 dict (실적 우선, 없으면 전체 폴백)
    df_perf = df[df['구분3'].str.strip() == '실적']
    perf_detail = df_perf.groupby(['G1', 'G2', '연도', '월'])['실적'].sum().to_dict()
    perf_group  = df_perf.groupby(['G1', '연도', '월'])['실적'].sum().to_dict()
    perf_total  = df_perf.groupby(['연도', '월'])['실적'].sum().to_dict()
    perf_plant  = df_perf.groupby(['G2', '연도', '월'])['실적'].sum().to_dict()

    all_detail = df.groupby(['G1', 'G2', '연도', '월'])['실적'].sum().to_dict()
    all_group  = df.groupby(['G1', '연도', '월'])['실적'].sum().to_dict()
    all_total  = df.groupby(['연도', '월'])['실적'].sum().to_dict()
    all_plant  = df.groupby(['G2', '연도', '월'])['실적'].sum().to_dict()

    def _get_with_fallback(key, d_perf, d_all):
        return float(d_perf[key]) if key in d_perf else float(d_all.get(key, 0.0))

    def val(g, p, yy, mm):           return _get_with_fallback((g, p, yy, mm), perf_detail, all_detail)
    def group_val(g, yy, mm):        return _get_with_fallback((g, yy, mm),    perf_group,  all_group)
    def grand_val(yy, mm):           return _get_with_fallback((yy, mm),       perf_total,  all_total)
    def plant_total_val(p, yy, mm):  return _get_with_fallback((p, yy, mm),    perf_plant,  all_plant)

    def months_for_avg(yy: int):
        upper = month if (yy == base_year and base_year == year) else 12
        return range(1, upper + 1)

    def year_avg(row_key: tuple, yy: int):
        key_g, key_p, kind = row_key
        vals = []
        for mm in months_for_avg(yy):
            if kind == 'plant':         vals.append(val(key_g, key_p, yy, mm))
            elif kind == 'group':       vals.append(group_val(key_g, yy, mm))
            elif kind == 'grand':       vals.append(grand_val(yy, mm))
            elif kind == 'plant_total': vals.append(plant_total_val(key_g, yy, mm))
        arr = np.array(vals, dtype=float)
        return float(arr.mean()) if arr.size else 0.0

    group_order = ["CHQ", "CD", "STS", "BTB", "PB"]
    plants_map  = {"CHQ": ["포항", "충주"], "CD": ["포항", "충주2"], "STS": ["포항", "충주2"], "BTB": [], "PB": []}

    rows = []
    for g in group_order:
        for p in plants_map[g]:
            rows.append(("", p))
        rows.append((g, ""))
    rows += [("합계", ""), ("포항", ""), ("충주", ""), ("충주2", "")]
    index = pd.MultiIndex.from_tuples(rows, names=["구분", ""])

    cols = [f"'{str(prev_year_for_avg)[-2:]}년 월평균",
            f"'{str(base_year)[-2:]}년 월평균"] + \
           [f"'{str(base_year)[-2:]}.{m}" for m in range(1, int(month) + 1)] + \
           ["전월대비", "%"]
    out = pd.DataFrame(0.0, index=index, columns=cols)

    prev_y, prev_m = (year, month - 1) if month > 1 else (year - 1, 12)

    r = 0
    for g in group_order:
        for p in plants_map[g]:
            out.iloc[r, out.columns.get_loc(f"'{str(prev_year_for_avg)[-2:]}년 월평균")] = year_avg((g, p, "plant"), prev_year_for_avg)
            out.iloc[r, out.columns.get_loc(f"'{str(base_year)[-2:]}년 월평균")]        = year_avg((g, p, "plant"), base_year)
            for mth in range(1, int(month) + 1):
                out.iloc[r, out.columns.get_loc(f"'{str(base_year)[-2:]}.{mth}")] = val(g, p, base_year, mth)
            curr, prev = val(g, p, year, month), val(g, p, prev_y, prev_m)
            diff = curr - prev
            out.iloc[r, out.columns.get_loc("전월대비")] = diff
            out.iloc[r, out.columns.get_loc("%")] = (diff / prev * 100.0) if prev != 0 else 0.0
            r += 1

        out.iloc[r, out.columns.get_loc(f"'{str(prev_year_for_avg)[-2:]}년 월평균")] = year_avg((g, "", "group"), prev_year_for_avg)
        out.iloc[r, out.columns.get_loc(f"'{str(base_year)[-2:]}년 월평균")]        = year_avg((g, "", "group"), base_year)
        for mth in range(1, int(month) + 1):
            out.iloc[r, out.columns.get_loc(f"'{str(base_year)[-2:]}.{mth}")] = group_val(g, base_year, mth)
        curr, prev = group_val(g, year, month), group_val(g, prev_y, prev_m)
        diff = curr - prev
        out.iloc[r, out.columns.get_loc("전월대비")] = diff
        out.iloc[r, out.columns.get_loc("%")] = (diff / prev * 100.0) if prev != 0 else 0.0
        r += 1

    out.iloc[r, out.columns.get_loc(f"'{str(prev_year_for_avg)[-2:]}년 월평균")] = year_avg(("합계", "", "grand"), prev_year_for_avg)
    out.iloc[r, out.columns.get_loc(f"'{str(base_year)[-2:]}년 월평균")]        = year_avg(("합계", "", "grand"), base_year)
    for mth in range(1, int(month) + 1):
        out.iloc[r, out.columns.get_loc(f"'{str(base_year)[-2:]}.{mth}")] = grand_val(base_year, mth)
    curr, prev = grand_val(year, month), grand_val(prev_y, prev_m)
    diff = curr - prev
    out.iloc[r, out.columns.get_loc("전월대비")] = diff
    out.iloc[r, out.columns.get_loc("%")] = (diff / prev * 100.0) if prev != 0 else 0.0
    r += 1

    for plant_total in ["포항", "충주", "충주2"]:
        out.iloc[r, out.columns.get_loc(f"'{str(prev_year_for_avg)[-2:]}년 월평균")] = year_avg((plant_total, "", "plant_total"), prev_year_for_avg)
        out.iloc[r, out.columns.get_loc(f"'{str(base_year)[-2:]}년 월평균")]        = year_avg((plant_total, "", "plant_total"), base_year)
        for mth in range(1, int(month) + 1):
            out.iloc[r, out.columns.get_loc(f"'{str(base_year)[-2:]}.{mth}")] = plant_total_val(plant_total, base_year, mth)
        curr, prev = plant_total_val(plant_total, year, month), plant_total_val(plant_total, prev_y, prev_m)
        diff = curr - prev
        out.iloc[r, out.columns.get_loc("전월대비")] = diff
        out.iloc[r, out.columns.get_loc("%")] = (diff / prev * 100.0) if prev != 0 else 0.0
        r += 1

    # 숫자형 정리
    num_cols = [c for c in out.columns if c != "%"]
    out[num_cols] = out[num_cols].round(0).astype(int)
    out["%"] = out["%"].astype(float)

    # === 표시 순서 강제 ===
    out = sort_board_like_item(out)

    return out

# =========================
# 부적합(포항) 요약 테이블
# =========================

# --- 공통: 순서 강제 유틸 ---
def order_like(df: pd.DataFrame, order: list[tuple]) -> pd.DataFrame:
    """
    df.index가 MultiIndex일 때, order에 적힌 튜플 순서대로 정렬.
    order에 없는 나머지 행은 기존 순서를 유지한 채 뒤에 붙인다.
    """
    if not isinstance(df.index, pd.MultiIndex):
        return df
    cur = list(df.index)
    seen = set()
    head = [t for t in order if t in cur and not (t in seen or seen.add(t))]
    tail = [t for t in cur if t not in set(head)]
    return df.reindex(head + tail)

# # 반올림
# return out.round(0)


def _build_defect_cols1(year: int, month: int) -> list[str]:
    prev   = f"{str(year-1)[-2:]}년 월평균"
    target = f"{str(year)[-2:]}년 목표"
    # ← 월에 '월' 글자 안 붙임
    months = [f"'{str(year)[-2:]}.{m}" for m in range(1, month+1)]
    return [prev, target] + months + ["합계", "월평균"]


def create_defect_summary_pohang(
    year: int,
    month: int,
    data: pd.DataFrame,
    months_window: tuple | None = None,
    plant_name: str = "포항"
) -> pd.DataFrame:
    """
    멀티인덱스(상/중/구분) 구조, 컬럼은 항상:
    [전년 월평균, 당년 목표, 1~선택월, 합계, 월평균]
    """
    df = data.copy()

    # ---------- Type Conversion/Normalization ----------
    for c in ['연도', '월', '실적']:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    for c in ['구분1', '구분2', '구분3', '구분4']:
        if c not in df.columns:
            df[c] = ''
        df[c] = df[c].fillna('').astype(str)

    # Filter for the target plant only
    df = df[df['구분1'].str.contains(plant_name)]
    prev_year = year - 1
    if months_window is None:
        months_window = tuple(range(1, month + 1))
    mlist = list(months_window)

    # ---------- Safe Sum/Mean Functions ----------
    safe_sum = lambda s: float(np.nansum(s)) if len(s) else 0.0
    safe_mean = lambda s: float(np.nanmean(s)) if len(s) else 0.0

    # ---------- Aggregation Helper ----------
    def pick(g2=None, g3=None, yy=None, mm=None, only_target=False):
        q = df
        if only_target:
            q = q[q['구분4'] == '목표']
        if g2 is not None:
            q = q[q['구분2'] == g2]
        if g3 is not None:
            q = q[q['구분3'] == g3]
        if yy is not None:
            q = q[q['연도'] == yy]
        if mm is not None:
            q = q[q['월'] == mm]
        return safe_sum(q['실적'])

    # ---------- Define Index/Columns (Internal level names are unique) ----------
    rows = [
        ('', '', '공정성'), ('', '', '소재성'), ('', 'CHQ', ''),
        ('', ' ', '공정성'), ('', ' ', '소재성'), ('', 'CD', ''),
        ('', '공정성', ''), ('', '소재성', ''), ('포항', '', ''),
    ]
    index = pd.MultiIndex.from_tuples(rows, names=['상', '중', '구분'])

    # cols = _build_defect_cols(year, month)
    # month_cols = [c for c in cols if c.endswith("월")]
    cols = _build_defect_cols1(year, month)
    prev_col, goal_col = cols[0], cols[1]
    month_cols = cols[2:-2]     # ← [1..선택월]을 라벨 무관하게 확보

    col_prev_avg = cols[0]
    col_target = cols[1]

    out = pd.DataFrame(0.0, index=index, columns=cols)

    # ---------- ① Previous Year Monthly Average ----------
    # CHQ
    chq_prev_ps = [pick(g2='CHQ', g3='공정성', yy=prev_year, mm=m) for m in range(1, 13)]
    chq_prev_ms = [pick(g2='CHQ', g3='소재성', yy=prev_year, mm=m) for m in range(1, 13)]
    out.loc[('', '', '공정성'), col_prev_avg] = safe_mean(chq_prev_ps)
    out.loc[('', '', '소재성'), col_prev_avg] = safe_mean(chq_prev_ms)
    out.loc[('', 'CHQ', ''), col_prev_avg] = safe_mean([a + b for a, b in zip(chq_prev_ps, chq_prev_ms)])

    # CD (Mid-level ' ' block)
    cd_prev_ps = [pick(g2='CD', g3='공정성', yy=prev_year, mm=m) for m in range(1, 13)]
    cd_prev_ms = [pick(g2='CD', g3='소재성', yy=prev_year, mm=m) for m in range(1, 13)]
    out.loc[('', ' ', '공정성'), col_prev_avg] = safe_mean(cd_prev_ps)
    out.loc[('', ' ', '소재성'), col_prev_avg] = safe_mean(cd_prev_ms)
    out.loc[('', 'CD', ''), col_prev_avg] = safe_mean([a + b for a, b in zip(cd_prev_ps, cd_prev_ms)])

    # Total/Pohang
    ps_all_prev = [pick(g3='공정성', yy=prev_year, mm=m) for m in range(1, 13)]
    ms_all_prev = [pick(g3='소재성', yy=prev_year, mm=m) for m in range(1, 13)]
    out.loc[('', '공정성', ''), col_prev_avg] = safe_mean(ps_all_prev)
    out.loc[('', '소재성', ''), col_prev_avg] = safe_mean(ms_all_prev)
    out.loc[('포항', '', ''), col_prev_avg] = safe_mean([ps_all_prev[i] + ms_all_prev[i] for i in range(12)])

    # ---------- ② Current Year Target (0 if not available) ----------
    out.loc[:, col_target] = 0.0

    # ---------- ③ Selected Months/Sum/Monthly Average ----------
    # CHQ
    chq_ps = [pick(g2='CHQ', g3='공정성', yy=year, mm=m) for m in mlist]
    chq_ms = [pick(g2='CHQ', g3='소재성', yy=year, mm=m) for m in mlist]
    out.loc[('', '', '공정성'), month_cols] = chq_ps
    out.loc[('', '', '소재성'), month_cols] = chq_ms
    out.loc[('', '', '공정성'), ['합계', '월평균']] = [safe_sum(chq_ps), safe_mean(chq_ps)]
    out.loc[('', '', '소재성'), ['합계', '월평균']] = [safe_sum(chq_ms), safe_mean(chq_ms)]
    out.loc[('', 'CHQ', ''), month_cols] = [chq_ps[i] + chq_ms[i] for i in range(len(mlist))]
    out.loc[('', 'CHQ', ''), ['합계', '월평균']] = [
        safe_sum(out.loc[('', 'CHQ', ''), month_cols]),
        safe_mean(out.loc[('', 'CHQ', ''), month_cols]),
    ]

    # CD
    cd_ps = [pick(g2='CD', g3='공정성', yy=year, mm=m) for m in mlist]
    cd_ms = [pick(g2='CD', g3='소재성', yy=year, mm=m) for m in mlist]
    out.loc[('', ' ', '공정성'), month_cols] = cd_ps
    out.loc[('', ' ', '소재성'), month_cols] = cd_ms
    out.loc[('', ' ', '공정성'), ['합계', '월평균']] = [safe_sum(cd_ps), safe_mean(cd_ps)]
    out.loc[('', ' ', '소재성'), ['합계', '월평균']] = [safe_sum(cd_ms), safe_mean(cd_ms)]
    out.loc[('', 'CD', ''), month_cols] = [cd_ps[i] + cd_ms[i] for i in range(len(mlist))]
    out.loc[('', 'CD', ''), ['합계', '월평균']] = [
        safe_sum(out.loc[('', 'CD', ''), month_cols]),
        safe_mean(out.loc[('', 'CD', ''), month_cols]),
    ]

    # Total Process/Material + Pohang Grand Total
    ps_all = [pick(g3='공정성', yy=year, mm=m) for m in mlist]
    ms_all = [pick(g3='소재성', yy=year, mm=m) for m in mlist]
    out.loc[('', '공정성', ''), month_cols] = ps_all
    out.loc[('', '소재성', ''), month_cols] = ms_all
    out.loc[('', '공정성', ''), ['합계', '월평균']] = [safe_sum(ps_all), safe_mean(ps_all)]
    out.loc[('', '소재성', ''), ['합계', '월평균']] = [safe_sum(ms_all), safe_mean(ms_all)]
    total = [ps_all[i] + ms_all[i] for i in range(len(mlist))]
    out.loc[('포항', '', ''), month_cols] = total
    out.loc[('포항', '', ''), ['합계', '월평균']] = [safe_sum(total), safe_mean(total)]

    # Ensure no missing columns/correct order
    out = out.reindex(columns=cols, fill_value=0).round(0)

    return out

# =========================
# 부적합(충주) 요약 테이블
# =========================

def create_defect_summary_chungju(
        year:int,
        month:int,
        data:pd.DataFrame,
        months_window:tuple,
        plant1_name:str="충주",      # 충주1공장
        plant2_name:str="충주2"      # 충주2공장 (CD만)
    ) -> pd.DataFrame:

        df = data.copy()

        # 숫자/문자 정리
        for c in ['연도','월','실적']:
            df[c] = pd.to_numeric(df.get(c), errors='coerce')
        for c in ['구분1','구분2','구분3','구분4']:
            if c not in df.columns: df[c] = ''
            df[c] = df[c].fillna('').astype(str)

        prev_year = year - 1
        mlist = list(months_window)

        # 안전 합/평균
        safe_sum  = lambda s: float(np.nansum(s))  if len(s) else 0.0
        safe_mean = lambda s: float(np.nanmean(s)) if len(s) else 0.0

        # 필터/집계
        def pick(plant=None, g2=None, cause=None, yy=None, mm=None, only_target=False):
            q = df.copy()
            if plant is not None:
                q = q[q['구분1'].str.contains(plant)]
            if only_target:
                q = q[q['구분4'] == '목표']
            if g2 is not None:
                q = q[q['구분2'] == g2]
            if cause is not None:
                q = q[q['구분3'] == cause]
            if yy is not None:
                q = q[q['연도'] == yy]
            if mm is not None:
                q = q[q['월'] == mm]
            return safe_sum(q['실적'])

        # ── 인덱스(9행) ──
        rows = [
            ('',   '',  '공정성'),
            ('',   '',  '소재성'),
            ('',  '충주1공장(CHQ)', ''),
            ('',   '',  '공정성'),
            ('',   '',  '소재성'),
            ('',   '충주2공장',      ''),
            ('','공정성',  ''),
            ('','소재성',  ''),
            ('충주','',    ''),
        ]
        index = pd.MultiIndex.from_tuples(rows, names=['상','중','하'])

        month_cols   = [f"'25.{m}" for m in mlist]
        col_prev_avg = f"{str(prev_year)[-2:]}년 월평균"
        col_target   = f"{str(year)[-2:]}년 목표"
        cols         = [col_prev_avg, col_target] + month_cols + ['합계','월평균']

        out = pd.DataFrame(0.0, index=index, columns=cols)

        # ---------- 전년 월평균 ----------
        # 충주1 CHQ (0,1,2)
        cj1_prev_ps = [pick(plant=plant1_name, g2='CHQ', cause='공정성', yy=prev_year, mm=m) for m in range(1,13)]
        cj1_prev_ms = [pick(plant=plant1_name, g2='CHQ', cause='소재성', yy=prev_year, mm=m) for m in range(1,13)]
        out.iloc[0, out.columns.get_loc(col_prev_avg)] = safe_mean(cj1_prev_ps)
        out.iloc[1, out.columns.get_loc(col_prev_avg)] = safe_mean(cj1_prev_ms)
        out.iloc[2, out.columns.get_loc(col_prev_avg)] = safe_mean([a+b for a,b in zip(cj1_prev_ps, cj1_prev_ms)])

        # 충주2 CD (3,4,5)
        cj2_prev_ps_cd = [pick(plant=plant2_name, g2='마봉강', cause='공정성', yy=prev_year, mm=m) for m in range(1,13)]
        cj2_prev_ms_cd = [pick(plant=plant2_name, g2='마봉강', cause='소재성', yy=prev_year, mm=m) for m in range(1,13)]
        out.iloc[3, out.columns.get_loc(col_prev_avg)] = safe_mean(cj2_prev_ps_cd)
        out.iloc[4, out.columns.get_loc(col_prev_avg)] = safe_mean(cj2_prev_ms_cd)
        out.iloc[5, out.columns.get_loc(col_prev_avg)] = safe_mean([a+b for a,b in zip(cj2_prev_ps_cd, cj2_prev_ms_cd)])

        # 전체 공정성/소재성/충주 총계 (6~8)
        chungju_mask = df['구분1'].str.contains(plant1_name) | df['구분1'].str.contains(plant2_name)
        def pick_chungju_total(yy, mm):
            q = df[(df['연도']==yy)&(df['월']==mm)&chungju_mask]
            return safe_sum(q['실적'])

        all_prev_ps = [pick(plant=None, g2=None, cause='공정성', yy=prev_year, mm=m) for m in range(1,13)]
        all_prev_ms = [pick(plant=None, g2=None, cause='소재성', yy=prev_year, mm=m) for m in range(1,13)]
        chungju_prev = [pick_chungju_total(prev_year, m) for m in range(1,13)]

        out.iloc[6, out.columns.get_loc(col_prev_avg)] = safe_mean(all_prev_ps)
        out.iloc[7, out.columns.get_loc(col_prev_avg)] = safe_mean(all_prev_ms)
        out.iloc[8, out.columns.get_loc(col_prev_avg)] = safe_mean(chungju_prev)

        # ---------- 목표(없으면 0) ----------
        out.loc[:, col_target] = 0.0

        # ---------- 당해 선택월 ----------
        # 충주1 CHQ (0~2)
        cj1_ps = [pick(plant=plant1_name, g2='CHQ', cause='공정성', yy=year, mm=m) for m in mlist]
        cj1_ms = [pick(plant=plant1_name, g2='CHQ', cause='소재성', yy=year, mm=m) for m in mlist]
        out.iloc[0, out.columns.get_indexer(month_cols)] = cj1_ps
        out.iloc[1, out.columns.get_indexer(month_cols)] = cj1_ms
        out.iloc[2, out.columns.get_indexer(month_cols)] = [cj1_ps[i]+cj1_ms[i] for i in range(len(mlist))]
        out.iloc[0, out.columns.get_indexer(['합계','월평균'])] = [safe_sum(cj1_ps), safe_mean(cj1_ps)]
        out.iloc[1, out.columns.get_indexer(['합계','월평균'])] = [safe_sum(cj1_ms), safe_mean(cj1_ms)]
        out.iloc[2, out.columns.get_indexer(['합계','월평균'])] = [
            safe_sum(out.iloc[2, out.columns.get_indexer(month_cols)]),
            safe_mean(out.iloc[2, out.columns.get_indexer(month_cols)]),
        ]

        # 충주2 CD (3~5)
        cj2_ps_cd = [pick(plant=plant2_name, g2='마봉강', cause='공정성', yy=year, mm=m) for m in mlist]
        cj2_ms_cd = [pick(plant=plant2_name, g2='마봉강', cause='소재성', yy=year, mm=m) for m in mlist]
        out.iloc[3, out.columns.get_indexer(month_cols)] = cj2_ps_cd
        out.iloc[4, out.columns.get_indexer(month_cols)] = cj2_ms_cd
        out.iloc[5, out.columns.get_indexer(month_cols)] = [cj2_ps_cd[i]+cj2_ms_cd[i] for i in range(len(mlist))]
        out.iloc[3, out.columns.get_indexer(['합계','월평균'])] = [safe_sum(cj2_ps_cd), safe_mean(cj2_ps_cd)]
        out.iloc[4, out.columns.get_indexer(['합계','월평균'])] = [safe_sum(cj2_ms_cd), safe_mean(cj2_ms_cd)]
        out.iloc[5, out.columns.get_indexer(['합계','월평균'])] = [
            safe_sum(out.iloc[5, out.columns.get_indexer(month_cols)]),
            safe_mean(out.iloc[5, out.columns.get_indexer(month_cols)]),
        ]

        # 전체 공정성/소재성/충주 총계 (6~8)
        ps_all = [pick(plant=None, g2=None, cause='공정성', yy=year, mm=m) for m in mlist]
        ms_all = [pick(plant=None, g2=None, cause='소재성', yy=year, mm=m) for m in mlist]
        chungju_total = [pick_chungju_total(year, m) for m in mlist]

        out.iloc[6, out.columns.get_indexer(month_cols)] = ps_all
        out.iloc[7, out.columns.get_indexer(month_cols)] = ms_all
        out.iloc[8, out.columns.get_indexer(month_cols)] = chungju_total

        for ridx in [6, 7, 8]:
            vals = out.iloc[ridx, out.columns.get_indexer(month_cols)].to_list()
            out.iloc[ridx, out.columns.get_indexer(['합계','월평균'])] = [safe_sum(vals), safe_mean(vals)]

        return out.round(0)


    # ---------------------------------------------
    # 표시 포맷(렌더용)
    # ---------------------------------------------
def format_total_production_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def fmt_int(x):
            try: return f"{int(x):,}"
            except Exception: return x

        def fmt_diff(x):
            try:
                xi = int(round(float(x)))
                return f"({abs(xi):,})" if xi < 0 else f"{xi:,}"
            except Exception:
                return x

        def fmt_pct(x):
            try: return f"{float(x):.1f}%"
            except Exception: return x

        for c in df.columns:
            if c == "전월대비": df[c] = df[c].apply(fmt_diff)
            elif c == "%":     df[c] = df[c].apply(fmt_pct)
            else:              df[c] = df[c].apply(fmt_int)

        return df.reset_index()


####### 비용분석


########################
##사용량 원단위 추이 포항##
########################

def create_material_usage_table_pohang(
    year: int,
    month: int,                 # 호출부는 그대로 사용
    data: pd.DataFrame,
    start_month: int = 2,
    plant_name: str = "포항",
    item_order: list[str] | None = None,
    round_digits: int = 6,
) -> pd.DataFrame:
    """
    [롱포맷 전용]
    - 필요한 컬럼: 구분3(공장), 구분1(항목), 월, 실적  (+ 선택: 연도)
    - 데이터에 존재하는 마지막 월(last_month)까지만 컬럼 생성
      -> end_month = min(month(호출값), last_month(데이터))
    - 열은 start_month ~ end_month 순서대로 배치
    - 행(item_order)은 교집합만 정렬하여 NaN 행 생성 방지
    """
    if start_month < 1 or start_month > 12:
        raise ValueError("start_month는 1~12 사이여야 합니다.")

    required = ["구분3", "구분1", "월", "실적"]
    df = data.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing} (필요: {required})")

    # 1) 공장/연도 필터
    q = df[df["구분3"].astype(str).str.contains(plant_name, na=False)].copy()
    if "연도" in q.columns:
        q = q[q["연도"] == year]

    # 2) 숫자화
    q["월"] = pd.to_numeric(q["월"], errors="coerce")
    q["실적"] = pd.to_numeric(q["실적"], errors="coerce")

    # 3) 데이터 기준 '마지막 월' 계산 + end_month 결정
    if q["월"].notna().any():
        last_month = int(q["월"].max())
    else:
        raise ValueError("유효한 월 값이 없습니다.")

    end_month = min(max(start_month, month), last_month)  # current_month를 데이터 마지막 월로 캡핑
    months_range = list(range(start_month, end_month + 1))

    # 4) 범위 필터 후 피벗
    q = q[q["월"].isin(months_range)]
    piv = q.pivot_table(index="구분1", columns="월", values="실적", aggfunc="sum")

    # 5) 컬럼명을 'n월'로 변경 + 순서 정렬 (누락 달은 NaN 열 추가)
    month_labels = [f"{m}월" for m in months_range]
    piv.columns = [f"{int(c)}월" for c in piv.columns]
    for c in month_labels:
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv[month_labels]

    # 6) 행 순서: 교집합만 정렬
    if item_order is None:
        item_order = ["열처리用LNG(㎥)", "질소(㎥)", "염산(kg)", "수소(㎥)", "산세用LNG(㎥)", "피막보급제(kg)"]
    order = [r for r in item_order if r in piv.index]
    if order:
        piv = piv.loc[order]

    # 7) 숫자/반올림 & 인덱스 라벨
    out = piv.apply(pd.to_numeric, errors="coerce").astype(float).round(round_digits)
    out.index.name = plant_name
    return out


########################
##사용량 원단위 추이 충주1##
########################

def create_material_usage_table_chungju1(
    year: int,
    month: int,                 # 호출부는 그대로 사용
    data: pd.DataFrame,
    start_month: int = 2,
    plant_name: str = "충주",
    item_order: list[str] | None = None,
    round_digits: int = 6,
) -> pd.DataFrame:
    """
    [롱포맷 전용]
    - 필요한 컬럼: 구분3(공장), 구분1(항목), 월, 실적  (+ 선택: 연도)
    - 데이터에 존재하는 마지막 월(last_month)까지만 컬럼 생성
      -> end_month = min(month(호출값), last_month(데이터))
    - 열은 start_month ~ end_month 순서대로 배치
    - 행(item_order)은 교집합만 정렬하여 NaN 행 생성 방지
    """
    if start_month < 1 or start_month > 12:
        raise ValueError("start_month는 1~12 사이여야 합니다.")

    required = ["구분3", "구분1", "월", "실적"]
    df = data.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing} (필요: {required})")

    # 1) 공장/연도 필터
    q = df[df["구분3"].astype(str).str.contains(plant_name, na=False)].copy()
    if "연도" in q.columns:
        q = q[q["연도"] == year]

    # 2) 숫자화
    q["월"] = pd.to_numeric(q["월"], errors="coerce")
    q["실적"] = pd.to_numeric(q["실적"], errors="coerce")

    # 3) 데이터 기준 '마지막 월' 계산 + end_month 결정
    if q["월"].notna().any():
        last_month = int(q["월"].max())
    else:
        raise ValueError("유효한 월 값이 없습니다.")

    end_month = min(max(start_month, month), last_month)  # current_month를 데이터 마지막 월로 캡핑
    months_range = list(range(start_month, end_month + 1))

    # 4) 범위 필터 후 피벗
    q = q[q["월"].isin(months_range)]
    piv = q.pivot_table(index="구분1", columns="월", values="실적", aggfunc="sum")

    # 5) 컬럼명을 'n월'로 변경 + 순서 정렬 (누락 달은 NaN 열 추가)
    month_labels = [f"{m}월" for m in months_range]
    piv.columns = [f"{int(c)}월" for c in piv.columns]
    for c in month_labels:
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv[month_labels]

    # 6) 행 순서: 교집합만 정렬
    if item_order is None:
        item_order = ["열처리用LNG(㎥)", "질소(㎥)", "염산(kg)", "수소(㎥)", "산세用LNG(㎥)", "피막보급제(kg)"]
    order = [r for r in item_order if r in piv.index]
    if order:
        piv = piv.loc[order]

    # 7) 숫자/반올림 & 인덱스 라벨
    out = piv.apply(pd.to_numeric, errors="coerce").astype(float).round(round_digits)
    out.index.name = plant_name
    return out

########################
##사용량 원단위 추이 충주2##
########################

def create_material_usage_table_chungju2(
    year: int,
    month: int,                 # 호출부는 그대로 사용
    data: pd.DataFrame,
    start_month: int = 2,
    plant_name: str = "충주2",
    item_order: list[str] | None = None,
    round_digits: int = 6,
) -> pd.DataFrame:
    """
    [롱포맷 전용]
    - 필요한 컬럼: 구분3(공장), 구분1(항목), 월, 실적  (+ 선택: 연도)
    - 데이터에 존재하는 마지막 월(last_month)까지만 컬럼 생성
      -> end_month = min(month(호출값), last_month(데이터))
    - 열은 start_month ~ end_month 순서대로 배치
    - 행(item_order)은 교집합만 정렬하여 NaN 행 생성 방지
    """
    if start_month < 1 or start_month > 12:
        raise ValueError("start_month는 1~12 사이여야 합니다.")

    required = ["구분3", "구분1", "월", "실적"]
    df = data.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing} (필요: {required})")

    # 1) 공장/연도 필터
    q = df[df["구분3"].astype(str).str.contains(plant_name, na=False)].copy()
    if "연도" in q.columns:
        q = q[q["연도"] == year]

    # 2) 숫자화
    q["월"] = pd.to_numeric(q["월"], errors="coerce")
    q["실적"] = pd.to_numeric(q["실적"], errors="coerce")

    # 3) 데이터 기준 '마지막 월' 계산 + end_month 결정
    if q["월"].notna().any():
        last_month = int(q["월"].max())
    else:
        raise ValueError("유효한 월 값이 없습니다.")

    end_month = min(max(start_month, month), last_month)  # current_month를 데이터 마지막 월로 캡핑
    months_range = list(range(start_month, end_month + 1))

    # 4) 범위 필터 후 피벗
    q = q[q["월"].isin(months_range)]
    piv = q.pivot_table(index="구분1", columns="월", values="실적", aggfunc="sum")

    # 5) 컬럼명을 'n월'로 변경 + 순서 정렬 (누락 달은 NaN 열 추가)
    month_labels = [f"{m}월" for m in months_range]
    piv.columns = [f"{int(c)}월" for c in piv.columns]
    for c in month_labels:
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv[month_labels]

    # 6) 행 순서: 교집합만 정렬
    if item_order is None:
        item_order = ["CD用SHOTBALL(kg)", "CD/BTB 방청유(Drum)"]
    order = [r for r in item_order if r in piv.index]
    if order:
        piv = piv.loc[order]

    # 7) 숫자/반올림 & 인덱스 라벨
    out = piv.apply(pd.to_numeric, errors="coerce").astype(float).round(round_digits)
    out.index.name = plant_name
    return out



#####단가추이#####

from typing import Optional, List

def _to_month_any(s) -> float:
    """문자/숫자에서 월(1~12) 추출: '2월','2025-02','02',' 5 ' 등 허용."""
    if pd.isna(s): return np.nan
    ss = str(s).strip()
    m = re.search(r"(\d{1,2})(?!.*\d)", ss) or re.search(r"(\d{1,2})", ss)
    if not m: return np.nan
    v = int(m.group(1))
    return float(v) if 1 <= v <= 12 else np.nan

def create_material_usage_table_unit_price(
    year: int,
    month: int,
    data: pd.DataFrame,
    start_month: int = 2,
    plant_name: Optional[str] = "충주",   # 필요없으면 None 전달하면 공장 필터 스킵
    item_order: Optional[List[str]] = None,
    round_digits: int = 6,
) -> pd.DataFrame:
    """
    [롱포맷 전용]
    - 필수 컬럼: 구분3(공장), 구분1(항목), 월, 실적  (+선택: 연도)
    - 대상 항목 6개만 필터 → 월별 합계 피벗
    - 열: start_month ~ 데이터의 마지막 월
    """
    if start_month < 1 or start_month > 12:
        raise ValueError("start_month는 1~12 사이여야 합니다.")

    # 기본 대상 항목 6개
    if item_order is None:
        item_order = [
            "LNG(천㎥, 전년동기)",
            "난방등유(L)",
            "LNG(천㎥)",
            "수소(천㎥)",
            "질소(천㎥)",
            "전력(천kwh)",
        ]

    # 원본 정리
    df = data.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 컬럼 존재 확인
    required = ["구분1", "월", "실적"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    q = df.copy()

    # 공장 필터 (원하면 plant_name=None로 스킵 가능)
    if plant_name is not None and "구분3" in q.columns:
        mask = q["구분3"].astype(str).str.contains(plant_name, na=False)
        q = q[mask] if mask.any() else q

    # 연도 필터(있을 때만; 비면 해제)
    if "연도" in q.columns:
        yr = pd.to_numeric(q["연도"], errors="coerce")
        q_year = q[yr == year]
        if len(q_year) > 0:
            q = q_year

    # 월/값 숫자화
    q["월"] = q["월"].apply(_to_month_any)
    q["실적"] = pd.to_numeric(q["실적"], errors="coerce")

    # 대상 6개만 남기기
    q = q[q["구분1"].isin(item_order)]

    # 유효 월만
    q = q[(q["월"] >= 1) & (q["월"] <= 12)]
    if q.empty:
        raise ValueError("유효한 월 데이터가 없습니다.")

    # 데이터의 마지막 월까지만 사용
    last_month = int(q["월"].max())
    end_month = min(max(start_month, month), last_month)
    months_range = list(range(start_month, end_month + 1))

    # 범위 필터 후 피벗(월별 합계)
    q = q[q["월"].isin(months_range)]
    piv = q.pivot_table(index="구분1", columns="월", values="실적", aggfunc="sum")

    # 컬럼명 'n월'로 통일 + 순서 보장
    piv.columns = [f"{int(c)}월" for c in piv.columns]
    month_labels = [f"{m}월" for m in months_range]
    for c in month_labels:
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv[month_labels]

    # 행 순서 고정(6개만, 없으면 NaN 행 생성)
    piv = piv.reindex(item_order)

    # 숫자/반올림 & 인덱스 라벨
    out = piv.apply(pd.to_numeric, errors="coerce").astype(float).round(round_digits)
    out.index.name = plant_name if plant_name is not None else ""
    return out



_NUM_PAT = re.compile(r"^\s*\((.*)\)\s*$")
def _to_number_robust(x)->float:
    if pd.isna(x): return 0.0
    s = str(x).replace("\u2212","-").replace(",","").strip()
    m = _NUM_PAT.match(s)
    if m: s = "-" + m.group(1).strip()
    try: return float(s)
    except: return 0.0

def load_nonop_cost_csv(source: str) -> pd.DataFrame:
    df = pd.read_csv(source, dtype=str)
    for c in ["구분1","구분2","구분3","구분4"]:
        if c in df.columns:
            #
            # 올바른 코드
            df[c] = (
                df[c]
                .astype(str)
                .fillna("")                 # 결측을 빈 문자열로
                .str.replace("\u00A0", " ", regex=False)  # 비정상 공백(NBSP) 제거(있을 때만)
                .str.strip()                # 앞뒤 공백 제거  ← 여기!
            )

    if "연도" in df: df["연도"] = pd.to_numeric(df["연도"].str.extract(r"(\d+)", expand=False), errors="coerce").fillna(0).astype(int)
    if "월"  in df: df["월"]  = pd.to_numeric(df["월"].str.extract(r"(\d+)",  expand=False), errors="coerce").fillna(0).astype(int)
    df["실적"] = df.get("실적", 0).apply(_to_number_robust)
    key = [c for c in ["구분1","구분2","구분3","구분4","연도","월"] if c in df.columns]
    return df.groupby(key, as_index=False, dropna=False)["실적"].sum()


def create_nonop_cost_3month_by_g2_g4(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:
    """
    섹션: 구분2 (기타비용/금융비용)
    행   : 구분4 (세부항목)
    특수 규칙:
      - 지급수수료(영업외) 아래에
          고철매각작업비 (실데이터)
          기타 = 잡손실 − 고철매각작업비
        를 'child'로 배치
      - 섹션 합계는 섹션 내 표시한 모든 행(부모/자식/일반)을 합산
    열   : 전전월/전월/당월 실적 + 증감(당월−전월)
    """
    y = int(year)
    m = max(int(month), 1)
    m1, m2 = max(m-1, 1), max(m-2, 1)

    df_y = data[(data["연도"] == y) & (data["월"].isin([m2, m1, m]))].copy()

    # 월 피벗 (구분2, 구분4)
    piv = (
        df_y.pivot_table(index=["구분2","구분4"], columns="월", values="실적",
                         aggfunc="sum", fill_value=0.0)
            .reindex(columns=[m2, m1, m], fill_value=0.0)
            .reset_index()
    )

    def _lbl(mm:int)->str: return f"'{str(y)[-2:]}.{mm}월 실적"
    c_m2, c_m1, c_m = _lbl(m2), _lbl(m1), _lbl(m)
    for mm, col in zip([m2, m1, m], [c_m2, c_m1, c_m]):
        piv[col] = piv[mm].astype(float)
    piv["증감"] = piv[c_m] - piv[c_m1]
    piv = piv.drop(columns=[m2, m1, m])

    # ── 섹션별(구분2) 원하는 노출 순서 ──
    order_fin = [  # 금융비용
        "이자비용",
        "외환차손",
        "외화환산손실",
        "파생상품평가손실",
        "파생상품거래손실",
        "매도가능증권손상차손",
        "리스부채이자비용",
        "기타금융부채평가손실",
    ]
    order_etc = [  # 기타비용
        "기부금",
        "유형자산처분손실",
        "무형자산처분손실",
        "무형자산손상차손",
        "지급수수료(영업외)",   # ↓ 이 아래에 child 2개
        "고철매각작업비",       # child (표시는 parent 아래에서만)
        "기타",                # child (= 잡손실 − 고철매각작업비)
        "잡손실",
        "공동지배기업투자처분손실",
        "종속기업주식손상차손",
        "기타비용",            # (구분4에 같은 이름이 있을 수 있음)
    ]
    

    rows = []

    def add_row(sec, acct, v2, v1, v, diff, row_type):
        rows.append({
            "구분": sec, "계정": acct,
            c_m2: float(v2), c_m1: float(v1), c_m: float(v), "증감": float(diff),
            "_row_type": row_type
        })

    # 섹션 처리 함수
    def build_section(sec_name: str, grp_df: pd.DataFrame, order_list: list[str]):
        # dict: 계정명 -> 레코드(행)
        recs = {r["구분4"]: r for r in grp_df.to_dict(orient="records")}
        start_idx = len(rows)

        for acct in order_list:
            if acct == "지급수수료(영업외)":
                # 부모 먼저
                parent = recs.get("지급수수료(영업외)")
                if parent is not None:
                    add_row("", "지급수수료(영업외)",
                            parent[c_m2], parent[c_m1], parent[c_m], parent["증감"], "parent")
                # child 1: 고철매각작업비 (실데이터가 있으면)
                steel = recs.get("고철매각작업비")
                # child 2: 기타 = 잡손실 − 고철매각작업비
                jab   = recs.get("잡손실")

                if steel is not None:
                    add_row("", "고철매각작업비",
                            steel[c_m2], steel[c_m1], steel[c_m], steel["증감"], "child")
                # 기타 = 잡손실 − 고철
                # (둘 중 하나 없으면 없는 값은 0으로 간주)
                v_j2 = float(jab[c_m2]) if jab is not None else 0.0
                v_j1 = float(jab[c_m1]) if jab is not None else 0.0
                v_j  = float(jab[c_m])  if jab is not None else 0.0
                d_j  = float(jab["증감"]) if jab is not None else 0.0

                v_s2 = float(steel[c_m2]) if steel is not None else 0.0
                v_s1 = float(steel[c_m1]) if steel is not None else 0.0
                v_s  = float(steel[c_m])  if steel is not None else 0.0
                d_s  = float(steel["증감"]) if steel is not None else 0.0

                add_row("", "기타", v_j2 - v_s2, v_j1 - v_s1, v_j - v_s, d_j - d_s, "child")

                # 이미 처리했으니 이후 루프에서 중복 추가 방지
                continue

            if acct in ("고철매각작업비", "기타"):
                # 위에서 parent 밑 child로 처리했으므로 스킵
                continue

            # 일반 계정
            rec = recs.get(acct)
            if rec is not None:
                add_row("", acct, rec[c_m2], rec[c_m1], rec[c_m], rec["증감"], "item")

        # 섹션 합계(이번 섹션에서 방금 추가한 모든 행 합산)
        sec_block = rows[start_idx:]
        if sec_block:
            sec_df = pd.DataFrame(sec_block)
            s = sec_df[[c_m2, c_m1, c_m, "증감"]].sum(numeric_only=True)
            add_row(sec_name, "", s[c_m2], s[c_m1], s[c_m], s["증감"], "section_total")

    # 섹션별 빌드
    preferred = ["기타비용", "금융비용"]
    others = [s for s in piv["구분2"].dropna().unique().tolist() if s not in preferred]
    sec_order = preferred + others  # ['기타비용','금융비용', ...]

    for sec in sec_order:
        grp = piv[piv["구분2"] == sec]
        if grp.empty:
            continue

        if sec == "기타비용":
            build_section("기타비용", grp, order_etc)   # (기부금, 유형자산처분손실, ... 순서 포함)
        elif sec == "금융비용":
            build_section("금융비용", grp, order_fin)   # (이자비용, 외환차손, ... 순서 포함)
        else:
            # 그 외 섹션은 구분4 사전순 또는 원하시는 고정 리스트를 추가로 지정 가능
            build_section(str(sec), grp, sorted(grp["구분4"].unique().tolist()))

    # 최종 '계'
    out = pd.DataFrame(rows)
    grand = out[out["_row_type"] == "section_total"][[c_m2, c_m1, c_m, "증감"]].sum(numeric_only=True)
    rows.append({
        "구분": "계", "계정": "",
        c_m2: float(grand[c_m2]) if not pd.isna(grand[c_m2]) else 0.0,
        c_m1: float(grand[c_m1]) if not pd.isna(grand[c_m1]) else 0.0,
        c_m:  float(grand[c_m])  if not pd.isna(grand[c_m])  else 0.0,
        "증감": float(grand["증감"]) if not pd.isna(grand["증감"]) else 0.0,
        "_row_type": "grand_total"
    })
    out = pd.DataFrame(rows)

    return out[["구분", "계정", c_m2, c_m1, c_m, "증감", "_row_type"]]


##### 실적 분석 #####


# ====== 손익 연결 ======
def _clean_profit_connected_df(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    # 기본 정리
    df['연도'] = df['연도'].astype(int)
    df['월'] = df['월'].astype(int)
    # "1,234,567" 형태 → 숫자
    df['실적'] = (
        df['실적']
        .astype(str)
        .str.replace(',', '', regex=False)
        .replace({'': None, 'nan': None})
        .astype(float)
    )

    # 스케일링
    money_metrics = ['매출액', '영업이익', '순금융비요', '경상이익']
    qty_metrics   = ['판매량']

    df.loc[df['구분3'].isin(money_metrics), '실적'] = df.loc[df['구분3'].isin(money_metrics), '실적'] / 1_000_000
    df.loc[df['구분3'].isin(qty_metrics),   '실적'] = df.loc[df['구분3'].isin(qty_metrics),   '실적'] / 1_000

    # 무의미한 열 제거(있다면)
    drop_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    return df


def create_connected_profit_table(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:

    df = _clean_profit_connected_df(data)

    companies = ['본사', '남통', '천진', '타이']
    metrics   = ['매출액', '판매량', '영업이익', '순금융비요', '경상이익']

    # 연간 계획(12개월 합), 전월 실적, 당월 계획/실적, 누적(1~month)
    def msum(f):
        return f.groupby(['구분2', '구분3'])['실적'].sum()

    # 연간 계획(해당 연도 1~12월 '계획' 합)
    plan_year = msum(df[(df['연도'] == year) & (df['구분4'] == '계획') & (df['월'].between(1, 12))])

    # 전월(전월 실적; month==1이면 0 처리)
    if month > 1:
        prev_actual = df[(df['연도'] == year) & (df['구분4'] == '실적') & (df['월'] == month - 1)] \
            .groupby(['구분2', '구분3'])['실적'].sum()
    else:
        prev_actual = pd.Series(0, index=plan_year.index)

    # 당월 계획/실적
    curr_plan = df[(df['연도'] == year) & (df['구분4'] == '계획') & (df['월'] == month)] \
        .groupby(['구분2', '구분3'])['실적'].sum()
    curr_actual = df[(df['연도'] == year) & (df['구분4'] == '실적') & (df['월'] == month)] \
        .groupby(['구분2', '구분3'])['실적'].sum()

    # 누적
    cum_plan = df[(df['연도'] == year) & (df['구분4'] == '계획') & (df['월'] <= month)] \
        .groupby(['구분2', '구분3'])['실적'].sum()
    cum_actual = df[(df['연도'] == year) & (df['구분4'] == '실적') & (df['월'] <= month)] \
        .groupby(['구분2', '구분3'])['실적'].sum()

    # (회사, 지표) 인덱스 뼈대
    idx = pd.MultiIndex.from_product([companies, metrics], names=['회사', '지표'])

    # 합산을 위한 helper
    def reidx(s):  # 누락키 0 보정
        s = s.reindex(idx, fill_value=0)
        return s

    # 본체 표 구성
    col_year_plan = f"'{str(year)[-2:]}년 계획"
    cols = [
        col_year_plan, '전월', '당월 계획', '당월 실적', '당월 계획대비', '당월 전월대비',
        '당월누적 계획', '당월누적 실적', '당월누적 계획대비'
    ]
    out = pd.DataFrame(index=idx, columns=cols, dtype=float)

    out[col_year_plan]      = reidx(plan_year).values
    out['전월']             = reidx(prev_actual).values
    out['당월 계획']        = reidx(curr_plan).values
    out['당월 실적']        = reidx(curr_actual).values
    out['당월 계획대비']     = out['당월 실적'] - out['당월 계획']
    out['당월 전월대비']     = out['당월 실적'] - out['전월']
    out['당월누적 계획']     = reidx(cum_plan).values
    out['당월누적 실적']     = reidx(cum_actual).values
    out['당월누적 계획대비']   = out['당월누적 실적'] - out['당월누적 계획']

    # ===== 합계 행 추가(회사=합계) =====
    sum_block = out.groupby(level='지표').sum(numeric_only=True)
    sum_block.index = pd.MultiIndex.from_product([['합계'], sum_block.index], names=['회사', '지표'])
    out = pd.concat([out, sum_block])

    # 보기 좋게 정렬(회사 순서, 지표 순서)
    order_idx = pd.MultiIndex.from_product([companies + ['합계'], metrics], names=['회사', '지표'])
    out = out.reindex(order_idx)

    # 숫자 0.0 → 0 처리
    out = out.fillna(0.0)

    return out



def _coerce_number_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(',', '', regex=False)
        .replace({'': None, 'nan': None})
        .astype(float)
    )

def _normalize_company_name(x: str) -> str:
    if x in ['타이', '태국']:
        return '태국'
    return x

def _clean_profit_connected_df_for_snapshot(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 표준화
    df['연도'] = df['연도'].astype(int)
    df['월'] = df['월'].astype(int)
    df['구분2'] = df['구분2'].astype(str).map(_normalize_company_name)
    df['구분3'] = df['구분3'].astype(str)
    df['구분4'] = df['구분4'].astype(str)
    df['실적'] = _coerce_number_series(df['실적'])

    # 스케일링
    money_metrics = ['매출액', '영업이익', '순금융비요', '경상이익']
    qty_metrics   = ['판매량']
    df.loc[df['구분3'].isin(money_metrics), '실적'] = df.loc[df['구분3'].isin(money_metrics), '실적'] / 1_000_000
    df.loc[df['구분3'].isin(qty_metrics),   '실적'] = df.loc[df['구분3'].isin(qty_metrics),   '실적'] / 1_000

    # 열 정리
    drop_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    return df

def _sum_at(df: pd.DataFrame, y: int, m: int, kind: str) -> pd.Series:
    """특정 연-월, 계획/실적(kind) 합계: (회사, 지표)로 그룹."""
    return (
        df[(df['연도'] == y) & (df['월'] == m) & (df['구분4'] == kind)]
        .groupby(['구분2', '구분3'])['실적'].sum()
    )

def _pp(val):
    """증감 숫자를 괄호 표기로(음수만 괄호)."""
    if pd.isna(val):
        return ""
    try:
        v = float(val)
    except Exception:
        return val
    s = f"{abs(int(round(v))):,}"
    return f"({s})" if v < 0 else s

def _fmt_int(val):
    try:
        return f"{int(round(float(val))):,}"
    except Exception:
        return val

def _fmt_pct(val):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return ""
    try:
        return f"{float(val):.1f}"
    except Exception:
        return val

def create_connected_profit_snapshot_table(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:
   
    df = _clean_profit_connected_df_for_snapshot(data)

    companies_order = ['본사', '중국', '남통', '천진', '태국']
    metrics_order   = ['매출액', '판매량', '영업이익', '%(영업)', '순금융비용', '경상이익', '%(경상)']

    # 전전월/전월/당월 인덱스 계산
    if month == 1:
        yy_prev = year - 1
        mm_prev = 12
        yy_prev2 = year - 1
        mm_prev2 = 11
    elif month == 2:
        yy_prev = year
        mm_prev = 1
        yy_prev2 = year - 1
        mm_prev2 = 12
    else:
        yy_prev = year
        mm_prev = month - 1
        yy_prev2 = year
        mm_prev2 = month - 2

    # 집계 시리즈(회사,지표)
    s_prev2 = _sum_at(df, yy_prev2, mm_prev2, '실적')
    s_prev  = _sum_at(df, yy_prev,  mm_prev,  '실적')
    s_plan  = _sum_at(df, year, month, '계획')
    s_curr  = _sum_at(df, year, month, '실적')

    # 합계(지표 기준)
    def total_by_metric(s):
        return s.groupby('구분3').sum()

    tot_prev2 = total_by_metric(s_prev2)
    tot_prev  = total_by_metric(s_prev)
    tot_plan  = total_by_metric(s_plan)
    tot_curr  = total_by_metric(s_curr)

    # 회사별 당월 실적 피벗(열=회사, 행=지표)
    company_curr = s_curr.reset_index().pivot_table(index='구분3', columns='구분2', values='실적', aggfunc='sum').reindex(columns=companies_order).fillna(0.0)

    # 퍼센트(합계, 회사별)
    def safe_ratio(n, d):
        return None if (d is None or d == 0 or pd.isna(d)) else n / d * 100

    op_margin_prev2 = safe_ratio(tot_prev2.get('영업이익', 0), tot_prev2.get('매출액', 0))
    op_margin_prev  = safe_ratio(tot_prev.get('영업이익', 0),  tot_prev.get('매출액', 0))
    op_margin_plan  = safe_ratio(tot_plan.get('영업이익', 0),  tot_plan.get('매출액', 0))
    op_margin_curr  = safe_ratio(tot_curr.get('영업이익', 0),  tot_curr.get('매출액', 0))

    or_margin_prev2 = safe_ratio(tot_prev2.get('경상이익', 0), tot_prev2.get('매출액', 0))
    or_margin_prev  = safe_ratio(tot_prev.get('경상이익', 0),  tot_prev.get('매출액', 0))
    or_margin_plan  = safe_ratio(tot_plan.get('경상이익', 0),  tot_plan.get('매출액', 0))
    or_margin_curr  = safe_ratio(tot_curr.get('경상이익', 0),  tot_curr.get('매출액', 0))

    # 회사별 퍼센트(당월)
    def company_margin_row(nm):
        # nm: '영업이익' 또는 '경상이익'
        num = company_curr.loc[nm] if nm in company_curr.index else pd.Series(0, index=company_curr.columns)
        den = company_curr.loc['매출액'] if '매출액' in company_curr.index else pd.Series(0, index=company_curr.columns)
        out = []
        for c in companies_order:
            v = None if (c not in num or c not in den or den[c] == 0) else num[c] / den[c] * 100
            out.append(v)
        return out

    comp_op_margin = company_margin_row('영업이익')
    comp_or_margin = company_margin_row('경상이익')

    # 표 본체 생성
    cols = ['전전월 실적', '전월 실적', '당월 계획', '당월 실적'] + companies_order + ['전월 실적 대비', '계획 대비']
    out = pd.DataFrame(index=metrics_order, columns=cols, dtype=object)

    # ─ 숫자 행 채우기 ─
    for metric in ['매출액', '판매량', '영업이익', '순금융비용', '경상이익']:
        out.at[metric, '전전월 실적'] = _fmt_int(tot_prev2.get(metric, 0))
        out.at[metric, '전월 실적']  = _fmt_int(tot_prev.get(metric, 0))
        out.at[metric, '당월 계획']  = _fmt_int(tot_plan.get(metric, 0))
        out.at[metric, '당월 실적']  = _fmt_int(tot_curr.get(metric, 0))
        # 회사별(당월 실적)
        for c in companies_order:
            v = company_curr.get(c).get(metric, 0) if metric in company_curr.index else 0
            out.at[metric, c] = _fmt_int(v)
        # 증감(합계 기준)
        diff_prev  = tot_curr.get(metric, 0) - tot_prev.get(metric, 0)
        diff_plan  = tot_curr.get(metric, 0) - tot_plan.get(metric, 0)
        out.at[metric, '전월 실적 대비'] = _pp(diff_prev)
        out.at[metric, '계획 대비']     = _pp(diff_plan)

    # ─ 퍼센트 행(영업이익/매출액, 경상이익/매출액) ─
    # 합계 4열
    out.at['%(영업)', '전전월 실적'] = _fmt_pct(op_margin_prev2)
    out.at['%(영업)', '전월 실적']  = _fmt_pct(op_margin_prev)
    out.at['%(영업)', '당월 계획']  = _fmt_pct(op_margin_plan)
    out.at['%(영업)', '당월 실적']  = _fmt_pct(op_margin_curr)

    out.at['%(경상)', '전전월 실적'] = _fmt_pct(or_margin_prev2)
    out.at['%(경상)', '전월 실적']  = _fmt_pct(or_margin_prev)
    out.at['%(경상)', '당월 계획']  = _fmt_pct(or_margin_plan)
    out.at['%(경상)', '당월 실적']  = _fmt_pct(or_margin_curr)

    # 회사별 5열(당월 기준)
    for i, c in enumerate(companies_order):
        out.at['%(영업)', c] = _fmt_pct(comp_op_margin[i])
        out.at['%(경상)', c] = _fmt_pct(comp_or_margin[i])

    # 증감(퍼센트포인트)
    def pp_delta(a, b):
        if a is None or b is None:
            return ""
        try:
            return _pp(a - b)  # 괄호표기는 음수만, 절대값 천단위
        except Exception:
            return ""

    out.at['%(영업)', '전월 실적 대비'] = pp_delta(op_margin_curr, op_margin_prev)
    out.at['%(영업)', '계획 대비']     = pp_delta(op_margin_curr, op_margin_plan)
    out.at['%(경상)', '전월 실적 대비'] = pp_delta(or_margin_curr, or_margin_prev)
    out.at['%(경상)', '계획 대비']     = pp_delta(or_margin_curr, or_margin_plan)

    out.index = ['매출액', '판매량', '영업이익', '%(영업)', '순금융비용', '경상이익', '%(경상)']
    return out



# ====================== 현금흐름표 ======================
import pandas as pd
import numpy as np

def _normalize_company_cf(x: str) -> str:
    x = str(x)
    if x in ("타이", "태국"):
        return "태국"
    return x

def _paren_to_signed(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.replace(",", "", regex=False).str.strip()
    neg = s.str.match(r"^\(.*\)$")
    s = s.str.replace(r"[\(\)]", "", regex=True)
    v = pd.to_numeric(s, errors="coerce")
    v[neg] = -v[neg].abs()
    return v.fillna(0.0)

def clean_cashflow_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 항목/회사/연월 컬럼 탐색은 기존과 동일...
    item_col = next((c for c in ["구분", "구분3", "항목"] if c in df.columns), None)
    if item_col is None:
        raise ValueError("CSV에 '구분'(또는 '구분3'/'항목') 컬럼이 필요합니다.")
    comp_col = next((c for c in ["구분2", "회사", "법인"] if c in df.columns), None)
    if comp_col is None:
        comp_col = "_회사"; df[comp_col] = "전체"
    if "연도" not in df.columns or "월" not in df.columns or "실적" not in df.columns:
        raise ValueError("CSV에 '연도','월','실적' 컬럼이 필요합니다.")

    # ▶ 문자열 정리(중복 방지용): 좌우공백/연속공백 제거
    df[item_col] = (
        df[item_col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    df[comp_col] = (
        df[comp_col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        .map(_normalize_company_cf)
    )

    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df["월"]   = pd.to_numeric(df["월"],   errors="coerce").astype("Int64")
    df["실적"] = _paren_to_signed(df["실적"])

    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    return df.rename(columns={item_col: "구분", comp_col: "회사"})


def create_cashflow_by_gubun(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:
    df = clean_cashflow_df(data)
    companies = ["본사", "남통", "천진", "태국"]

    # 선택 월 데이터 없으면 최근 과거월로 폴백
    avail = sorted(df.loc[df["연도"] == year, "월"].dropna().unique())
    used_month = month
    if len(avail) and month not in avail:
        past = [m for m in avail if m <= month]
        used_month = int(max(past) if past else max(avail))

    # 파일 등장 순서(중복 제거, 순서 보존)
    gubun_order = list(dict.fromkeys(df["구분"].astype(str).tolist()))

    # ---------- 집계 함수 ----------
    def total_by_items(y, months):
        q = (df["연도"] == y) & (df["월"].isin(months))
        s = (
            df[q]
            .groupby("구분", sort=False)["실적"]
            .sum()                         # 1차 집계
        )
        # 
        if s.index.duplicated().any():
            s = s.groupby(level=0).sum()
        return s

    def company_by_month(y, m):
        q = (df["연도"] == y) & (df["월"] == m)
        pv = (
            df[q]
            .pivot_table(index="구분", columns="회사", values="실적",
                         aggfunc="sum", fill_value=0.0, observed=False)
        )
        # ▶ 중복 방지: 인덱스 중복 시 합산
        if pv.index.duplicated().any():
            pv = pv.groupby(level=0).sum()
        # 원하는 열만, 순서대로
        pv = pv.reindex(columns=companies).fillna(0.0)
        return pv
    # ---------- /집계 함수 ----------

    col_24      = total_by_items(year - 1, range(1, 13))
    col_25_prev = total_by_items(year, range(1, used_month)) if used_month > 1 else col_24 * 0
    col_month   = total_by_items(year, [used_month])
    col_ytd     = total_by_items(year, range(1, used_month + 1))
    by_comp     = company_by_month(year, used_month)

    # ▶ 출력 인덱스: 중복 제거된 순서 리스트
    all_items = pd.Index(gubun_order, name="구분")

    out = pd.DataFrame(index=all_items, dtype=float)
    out["'24"]      = col_24.reindex(all_items).fillna(0.0).values
    out["'25"]      = col_25_prev.reindex(all_items).fillna(0.0).values
    out["당월"]      = col_month.reindex(all_items).fillna(0.0).values
    out["당월누적"]   = col_ytd.reindex(all_items).fillna(0.0).values

    # 법인별 당월
    for c in companies:
        # by_comp는 인덱스 유일화 완료 상태
        out[c] = by_comp.reindex(all_items).get(c, 0.0).fillna(0.0).values

    # 최종 컬럼 순서
    out = out[["'24", "'25", "당월", "본사", "남통", "천진", "태국", "당월누적"]]
    return out


# ====================== 재무상태표 ======================


import pandas as pd
import numpy as np

def _bs_to_number(x):
    s = str(x).strip()
    if not s:
        return 0.0
    neg = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace(',', '')
    try:
        v = float(s)
    except Exception:
        return 0.0
    return -abs(v) if neg else v


import pandas as pd
import numpy as np

def _bs_to_number(x):
    s = str(x).strip()
    if not s:
        return 0.0
    neg = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace(',', '')
    try:
        v = float(s)
    except Exception:
        return 0.0
    return -abs(v) if neg else v

def _normalize_bs_simple(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    item_col = next((c for c in ['구분3','구분2','항목','구분'] if c in df.columns), None)
    comp_col = next((c for c in ['회사','법인','구분4'] if c in df.columns), None)

    for need in ['연도','월','실적']:
        if need not in df.columns:
            raise ValueError(f"재무상태표 데이터에 '{need}' 컬럼이 필요합니다.")
    if item_col is None:
        raise ValueError("재무상태표: '구분3/구분2/항목/구분' 중 하나가 필요합니다.")

    df[item_col] = df[item_col].astype(str).str.strip()
    if comp_col is None:
        comp_col = '_회사'
        df[comp_col] = '전체'
    else:
        df[comp_col] = df[comp_col].astype(str).str.strip().replace({'타이':'태국'})

    s = df['실적'].astype(str).str.strip()
    neg = s.str.match(r'^\(.*\)$')
    s = s.str.replace(r'[(),]', '', regex=True)
    v = pd.to_numeric(s, errors='coerce').fillna(0.0)
    v[neg] = -v[neg].abs()
    df['실적'] = v

    df['연도'] = pd.to_numeric(df['연도'], errors='coerce').astype('Int64')
    df['월']   = pd.to_numeric(df['월'],   errors='coerce').astype('Int64')

    drop_cols = [c for c in df.columns if str(c).startswith('Unnamed')]
    if drop_cols: df = df.drop(columns=drop_cols, errors='ignore')

    return df.rename(columns={item_col:'구분3', comp_col:'회사'})

def _closest_month(df: pd.DataFrame, year: int, month: int) -> int | None:
    avail = sorted(m for m in df.loc[df['연도']==year, '월'].dropna().unique())
    if not avail: return None
    le = [m for m in avail if m <= month]
    return le[-1] if le else avail[-1]

def create_bs_by_items(year:int, month:int, data:pd.DataFrame,
                                item_order:list[str]) -> pd.DataFrame:
    """
    구분3만으로 집계. item_order 순서대로 행을 만들고, 회사별(현재월) 열도 함께 생성.
    반환: index='구분', columns=["'24","'25","당월", <회사...>, "전월비 증감"]
    """
    df = _normalize_bs_simple(data)

    used_m = _closest_month(df, year, month) or _closest_month(df, year, 12)
    if used_m and used_m > 1:
        prev_y, prev_m = year, used_m - 1
        if _closest_month(df, prev_y, prev_m) is None:
            prev_m = _closest_month(df, year, used_m - 1)
    else:
        prev_y, prev_m = year - 1, _closest_month(df, year - 1, 12)
    last_prev_year_m = _closest_month(df, year - 1, 12)

    # 현재월에 실제로 존재하는 회사들로 열 구성
    comp_exists = sorted(df.loc[(df['연도']==year) & (df['월']==used_m), '구분2'].dropna().unique())####
    prefer = ['특수강','본사','남통','천진','태국']
    comp_cols = [c for c in prefer if c in comp_exists] + [c for c in comp_exists if c not in prefer]

    rows = []
    for item in item_order:
        mask_item = df['구분3'] == item

        def _sum_at(y, m):
            if y is None or m is None: return 0.0
            return float(df[mask_item & (df['연도']==y) & (df['월']==m)]['실적'].sum())

        def _by_company(y, m):
            if y is None or m is None: return {c:0.0 for c in comp_cols}
            sub = df[mask_item & (df['연도']==y) & (df['월']==m)]
            s = sub.groupby('구분2')['실적'].sum()
            return {c: float(s.get(c, 0.0)) for c in comp_cols}

        v24 = _sum_at(year-1, last_prev_year_m)
        v25 = _sum_at(prev_y, prev_m)
        vm  = _sum_at(year,   used_m)
        comp_v = _by_company(year, used_m)

        row = {"'24": v24, "'25": v25, "당월": vm, **comp_v, "전월비 증감": vm - v25}
        rows.append(row)

    out = pd.DataFrame(rows, index=pd.Index(item_order, name='구분')).fillna(0.0)

    # 보기 좋은 열 순서
    comp_cols = [c for c in out.columns if c not in ["'24","'25","당월","전월비 증감"]]
    out = out[["'24","'25","당월"] + comp_cols + ["전월비 증감"]]


    out.attrs['used_month'] = used_m
    out.attrs['prev_month'] = prev_m
    return out

# ===== 회전일 =====

def _num_paren(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    neg = s.str.match(r'^\(.*\)$')
    s = s.str.replace(r'[(),]', '', regex=True)
    v = pd.to_numeric(s, errors='coerce')
    v[neg] = -v[neg].abs()
    return v


def _normalize_turnover(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    회전일 원천을 표준 스키마로 정리:
      - 항목: '구분' 컬럼로 반환
      - 회사: '회사' 컬럼로 반환
      - 값 : '값'  컬럼로 반환
    회사명이 구분3에 있는 형태(계/특수강/남(통)/천진/태국)도 자동 감지해서
    '회사'로 사용하고, 항목은 구분2(또는 구분/항목)를 사용한다.
    """
    df = df_raw.copy()

    # 값/항목/회사 후보
    val_col = next((c for c in ['회전일','실적','값','value'] if c in df.columns), None)
    if val_col is None:
        raise ValueError("회전일 데이터에 '회전일/실적/값' 컬럼이 필요합니다.")

    # 기본 숫자/연월 정리
    df['연도'] = pd.to_numeric(df['연도'], errors='coerce').astype('Int64')
    df['월']   = pd.to_numeric(df['월'],   errors='coerce').astype('Int64')

    # 괄호 음수 → 음수, 쉼표 제거
    df[val_col] = _num_paren(df[val_col]).astype(float)

    # 후보 열들
    cand_item = ['구분3','구분2','구분','항목']
    cand_comp = ['회사','법인','구분4']

    # 회사/항목 기본 선택
    item_col = next((c for c in cand_item if c in df.columns), None)
    comp_col = next((c for c in cand_comp if c in df.columns), None)

    # --- 회사명이 구분3에 들어있는 케이스 감지 ---
    known_companies = {'계','전체','연결','특수강','본사','남통','남동','천진','태국','타이'}
    def looks_like_company(series) -> bool:
        s = series.astype(str).str.strip()

        nonnull = s[s != ""]
        if len(nonnull) == 0: return False
        hit = nonnull.isin(known_companies).mean()
        return hit >= 0.7


    if ('구분3' in df.columns) and (comp_col is None or looks_like_company(df['구분3'])):
        comp_col = '구분3'
        # 항목은 구분2 > 구분 > 항목 순으로
        item_col = next((c for c in ['구분2','구분','항목'] if c in df.columns and c != comp_col), item_col)

    if item_col is None:
        raise ValueError("회전일 데이터에 항목 컬럼(구분3/구분2/구분/항목) 중 하나가 필요합니다.")

    # 문자열 정리 + alias
    df[item_col] = df[item_col].astype(str).str.strip()
    if comp_col is None:
        comp_col = '_회사'
        df[comp_col] = '전체'
    df[comp_col] = (
        df[comp_col].astype(str).str.strip()
          .replace({'타이':'태국', '남동':'남통'})   # alias 통일
    )

    # 쓰지 않는 Unnamed 열 제거
    drop_cols = [c for c in df.columns if str(c).startswith('Unnamed')]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    return df.rename(columns={item_col: '구분', comp_col: '회사', val_col: '값'})




def create_turnover(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:


    df = _normalize_turnover(data)   

    # ───────── 월 선택(선택 월이 없으면 가장 가까운 과거 월) ─────────
    avail_this = sorted(df.loc[df['연도'] == year, '월'].dropna().unique())
    used_m = month
    if avail_this and month not in avail_this:
        prior = [m for m in avail_this if m <= month]
        used_m = prior[-1] if prior else avail_this[-1]

    # 전월
    prev_y, prev_m = year, used_m - 1
    if prev_m < 1:
        prev_y = year - 1
        prev_avail = sorted(df.loc[df['연도'] == prev_y, '월'].dropna().unique())
        prev_m = prev_avail[-1] if prev_avail else 12

    # ───────── 회사 라벨 정규화 & 목록 생성 ─────────
    def norm_comp(x: str) -> str:
        x = str(x).strip()
        x = {'타이': '태국', '남동': '남통'}.get(x, x)  # 표기 통일
        if x in ('전체', '연결'):                      # 전체/연결 → 계
            return '계'
        return x

    # 사용월에 실제 존재하는 회사 라벨(정규화 후)만 채택
    exist_raw = df.loc[(df['연도'] == year) & (df['월'] == used_m), '회사'].dropna().astype(str)
    exist_norm = pd.Index(exist_raw.map(norm_comp).unique())

    prefer = ['특수강', '본사', '남통', '천진', '태국']
    # '계'와 빈값 제외 + 선호순 → 기타
    companies = [c for c in prefer if c in exist_norm] + [c for c in exist_norm if c not in prefer and c not in ('계', '')]

    # ───────── 항목×월 집계 도우미(정규화된 회사 기준) ─────────
    def row_by_month(item, y, m):
        sub = df[(df['구분'] == item) & (df['연도'] == y) & (df['월'] == m)].copy()
        if sub.empty:
            # 회사별 NaN
            return {'계': np.nan, **{c: np.nan for c in companies}}

        sub['회사N'] = sub['회사'].map(norm_comp)
        byc = sub.groupby('회사N', dropna=True)['값'].mean()  # 회전일은 평균 사용

        # 개별 회사 채우기
        res = {c: float(byc.get(c, np.nan)) for c in companies}

        # '계' 값: 데이터에 '계'가 있으면 사용, 없으면 개별 회사 평균
        if '계' in byc.index:
            res['계'] = float(byc['계'])
        else:
            vals = [res[c] for c in companies if pd.notnull(res[c])]
            res['계'] = float(np.mean(vals)) if vals else np.nan
        return res

    items = ['매출채권', '재고자산', '매임채무']

    rows_curr = {it: row_by_month(it, year, used_m) for it in items}
    rows_prev = {it: row_by_month(it, prev_y, prev_m) for it in items}

    # 현금전환주기 = 매출채권 + 재고자산 - 매입채무
    def combine_ccc(dic_a, dic_b, dic_c):
        keys = set(dic_a.keys()) | set(dic_b.keys()) | set(dic_c.keys())
        out = {}
        for k in keys:
            a = dic_a.get(k, np.nan)
            b = dic_b.get(k, np.nan)
            c = dic_c.get(k, np.nan)
            out[k] = (a if pd.notnull(a) else 0) + (b if pd.notnull(b) else 0) - (c if pd.notnull(c) else 0)
        return out

    curr_ccc = combine_ccc(rows_curr['매출채권'], rows_curr['재고자산'], rows_curr['매임채무'])
    prev_ccc = combine_ccc(rows_prev['매출채권'], rows_prev['재고자산'], rows_prev['매임채무'])

    # ───────── 표 구성 ─────────
    subcols = ['계'] + companies  # 중복/빈값 제거, 정규화된 최종 목록
    arrays = (['당월'] * len(subcols) + ['전월비'] * len(subcols),
              subcols + subcols)
    cols = pd.MultiIndex.from_arrays(arrays, names=['', ''])

    index_order = items + ['현금전환주기']
    out = pd.DataFrame(index=index_order, columns=cols, dtype=float)

    # 채우기
    def fill_row(it):
        cur = rows_curr[it]
        prv = rows_prev[it]
        for k in subcols:
            cur_v = cur.get(k, np.nan)
            prv_v = prv.get(k, np.nan)
            out.loc[it, ('당월', k)] = cur_v
            out.loc[it, ('전월비', k)] = (cur_v - prv_v) if (pd.notnull(cur_v) and pd.notnull(prv_v)) else np.nan

    for it in items:
        fill_row(it)

    # 현금전환주기
    for k in subcols:
        cv = curr_ccc.get(k, np.nan)
        pv = prev_ccc.get(k, np.nan)
        out.loc['현금전환주기', ('당월', k)] = cv
        out.loc['현금전환주기', ('전월비', k)] = (cv - pv) if (pd.notnull(cv) and pd.notnull(pv)) else np.nan

    # 표시용 반올림
    out = out.astype(float).round(1)

    # 헤더 라벨용 메타
    out.attrs['used_month'] = used_m
    out.attrs['prev_month'] = prev_m
    return out

##### ROE #####

_ZWS = "\u200b"          # 제로폭 공백
_WS_RE = re.compile(r"\s+", re.UNICODE)

def _clean_str(s):
    """제로폭/비정상 공백 제거 + strip + 내부 공백 1칸화"""
    if pd.isna(s): return ""
    t = str(s).replace(_ZWS, "")
    t = t.replace("\u00a0", " ")  # NBSP
    t = _WS_RE.sub(" ", t).strip()
    return t

def _norm_kind_label(x):
    """구분3 라벨 통일: 경영계획→계획, sps→SPS 등"""
    t = _clean_str(x).lower()
    if t in ("경영계획","계획","plan"):   return "계획"
    if t in ("수정계획","replan"):       return "수정계획"
    if t in ("sps","sps (100점)","sps(100점)"): return "SPS"
    if t in ("실적","actual"):           return "실적"
    return _clean_str(x)  # 기타는 원형 보존

def _norm_scope_label(x):
    """구분4 라벨 통일: 누적/YTD 만 사용"""
    t = _clean_str(x).lower()
    if "누적" in t or "ytd" in t:
        return "누적"
    return _clean_str(x)

def _to_year_col(s: pd.Series) -> pd.Series:
    def _f(x):
        t = _clean_str(x)
        m = re.search(r"\d{2,4}", t)
        if not m: return pd.NA
        v = m.group()
        if len(v) == 2: v = "20"+v
        try: return int(v)
        except: return pd.NA
    return s.map(_f).astype("Int64")

def _to_month_col(s: pd.Series) -> pd.Series:
    def _f(x):
        t = _clean_str(x)
        m = re.search(r"\d{1,2}", t)
        return int(m.group()) if m else pd.NA
    return s.map(_f).astype("Int64")

def _to_number(s: pd.Series) -> pd.Series:
    def _f(x):
        if pd.isna(x): return np.nan
        t = _clean_str(x).replace(",", "")
        neg = t.startswith("(") and t.endswith(")")
        t = t.strip("()").replace("%","")
        try:
            v = float(t)
            return -abs(v) if neg else v
        except:
            return np.nan
    return s.map(_f)

# ==== ROE 정규화 ====
def _normalize_roe_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    req = ['구분1','구분2','구분3','구분4','연도','월','실적']
    miss = [c for c in req if c not in df_raw.columns]
    if miss:
        raise ValueError(f"ROE 데이터에 필요한 컬럼 누락: {miss}")

    df = df_raw.copy()

    # 문자열/숫자 정리
    for c in ['구분1','구분2','구분3','구분4']:
        df[c] = df[c].map(_clean_str)

    df['연도'] = _to_year_col(df['연도'])
    df['월']   = _to_month_col(df['월'])
    df['실적'] = _to_number(df['실적'])

    # 대상: 구분1에 'ROE' 포함(연결), 구분4=누적만
    df = df[df['구분1'].str.contains("ROE", na=False)]
    df['구분3'] = df['구분3'].map(_norm_kind_label)
    df['구분4'] = df['구분4'].map(_norm_scope_label)
    df = df[df['구분4'].eq("누적")]

    return df

def create_roe_table(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_roe_table(data)

    # 당해 가용월 중 선택월 보정
    avail = sorted(df.loc[df['연도'].eq(year),'월'].dropna().unique())
    used_m = int(month)
    if avail and used_m not in avail:
        le = [m for m in avail if m <= used_m]
        used_m = int(le[-1] if le else avail[-1])

    prev_year, prev_month = year-1, 12

    # 안전 집계 도우미(여러 행 시: ROE=평균, 당기순=합계)
    def pick(metric: str, kind: str, y: int, m: int|None):
        sub = df[(df['구분2'].eq(metric)) & (df['구분3'].eq(kind)) & (df['연도'].eq(y))]
        if m is not None:
            mm = sub['월'].dropna().astype(int)
            if len(mm)==0: return np.nan
            if m in mm.values:
                sub = sub[ sub['월'].eq(m) ]
            else:
                le = mm[mm.le(m)]
                if len(le): sub = sub[ sub['월'].eq(int(le.max())) ]
                else: return np.nan
        else:
            # 계획류: 그 해 가장 늦은 월
            mm = sub['월'].dropna().astype(int)
            if len(mm)==0: return np.nan
            sub = sub[ sub['월'].eq(int(mm.max())) ]

        vals = sub['실적']
        if metric == 'ROE':
            return float(vals.mean()) if len(vals) else np.nan
        return float(vals.sum()) if len(vals) else np.nan

    cols = ["'"+str(prev_year)[-2:]+"년", f"{used_m}월 누적", "경영계획", "수정계획", "SPS (100점)"]
    out  = pd.DataFrame(index=['ROE*','당기순이익*'], columns=cols, dtype=float)

    # ROE
    out.loc['ROE*', cols[0]] = pick('ROE', '실적',   prev_year, prev_month)
    out.loc['ROE*', cols[1]] = pick('ROE', '실적',   year,      used_m)
    out.loc['ROE*', cols[2]] = pick('ROE', '계획',   year,      None)
    out.loc['ROE*', cols[3]] = pick('ROE', '수정계획', year,     None)
    out.loc['ROE*', cols[4]] = pick('ROE', 'SPS',    year,      None)

    # 당기순이익
    out.loc['당기순이익*', cols[0]] = pick('당기순이익','실적',   prev_year, prev_month)
    out.loc['당기순이익*', cols[1]] = pick('당기순이익','실적',   year,      used_m)
    out.loc['당기순이익*', cols[2]] = pick('당기순이익','계획',   year,      None)
    out.loc['당기순이익*', cols[3]] = pick('당기순이익','수정계획', year,     None)
    out.loc['당기순이익*', cols[4]] = pick('당기순이익','SPS',    year,      None)

    out.attrs['used_month'] = used_m
    return out


## 손익 별도
from decimal import Decimal, ROUND_HALF_UP

# ─────────────────────────────────────
# 공통 유틸: 반올림(HALF_UP) + 포맷 + 안전계산
# ─────────────────────────────────────
def _halfup0(x):
    """정수 자리 HALF_UP"""
    try:
        return int(Decimal(str(float(x))).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    except Exception:
        return None

def _halfup1(x):
    """소수 1자리 HALF_UP"""
    try:
        return float(Decimal(str(float(x))).quantize(Decimal('0.0'), rounding=ROUND_HALF_UP))
    except Exception:
        return None

def _num_coerce_pl(x):
    """숫자 파싱: 괄호=음수, 콤마 제거"""
    s = str(x).strip() if x is not None else ""
    if s == "" or s.lower() == "nan":
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "").replace(",", "")
    try:
        v = float(s)
        return -abs(v) if neg else v
    except:
        return np.nan

def _coerce_series_pl(s: pd.Series) -> pd.Series:
    return s.map(_num_coerce_pl)

def _norm_label_pl(s: str) -> str:
    if s is None: return ""
    t = str(s).replace("\u3000"," ").strip()
    return t.replace("순금융비요","순금융비용")  

def _ratio(n, d):
    """n/d*100 (둘 다 유효할 때만)"""
    try:
        n = float(n); d = float(d)
        if d == 0 or np.isnan(n) or np.isnan(d):
            return None
        return n/d*100
    except Exception:
        return None

def _diff(a, b):
    """a-b (둘 다 유효할 때만)"""
    try:
        if a is None or b is None: return None
        if isinstance(a, float) and np.isnan(a): return None
        if isinstance(b, float) and np.isnan(b): return None
        return float(a) - float(b)
    except Exception:
        return None

def _fmt_int(v):
    """백만원/천톤 등 정수표시(HALF_UP)"""
    r = _halfup0(v)
    return "" if r is None else f"{r:,}"

def _fmt_pp0(v):
    """증감(정수, 음수만 괄호)"""
    r = _halfup0(v)
    if r is None: return ""
    s = f"{abs(r):,}"
    return f"({s})" if r < 0 else s

def _fmt_pct(v):
    """% 값 1자리(HALF_UP) → '3.5' 형태(기호는 UI에서 묶을 수도)"""
    r = _halfup1(v)
    return "" if r is None else f"{r:.1f}"

def _fmt_pp1(v):
    """증감(퍼센트포인트, 1자리, 음수만 괄호)"""
    r = _halfup1(v)
    if r is None: return ""
    s = f"{abs(r):.1f}"
    return f"({s})" if r < 0 else s




def _sum_metric(df: pd.DataFrame, y: int, m: int, tag: str) -> pd.Series:
    sub = df[(df["연도"]==y) & (df["월"]==m) & (df["구분4"]==tag)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("구분3")["실적"].sum()

def _sum_metric_cum(df: pd.DataFrame, y: int, m: int, tag: str) -> pd.Series:
    sub = df[(df["연도"]==y) & (df["월"]<=m) & (df["구분4"]==tag)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("구분3")["실적"].sum()


def create_pl_separate_hq_snapshot(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:
    """
    손익(별도) 표 생성 (고정 스케일 방식).
    - 금액 지표(매출액/영업이익/순금융비용/경상이익)는 '원' 기준이라고 가정하고 여기서 /1,000,000 → '백만원'
    - 판매량은 '톤' 그대로
    - 퍼센트는 (이익/매출액)*100
    """
    import numpy as np
    import pandas as pd

    money_metrics = ["매출액", "영업이익", "순금융비용", "경상이익"]

    # ── 1) 최소 전처리: 라벨 정리/숫자 파싱/본사 필터(스케일링은 하지 않음) ──
    df = data.copy()

    for c in ["구분1","구분2","구분3","구분4"]:
        if c not in df.columns: df[c] = ""
        df[c] = (
            df[c].astype(str)
                 .str.replace("\u3000"," ")
                 .str.strip()
                 .replace({"순금융비요":"순금융비용"})
        )

    # 본사만
    df = df[df["구분2"] == "본사"].copy()

    # 연/월
    df["연도"] = pd.to_numeric(df.get("연도"), errors="coerce").astype("Int64")
    df["월"]   = pd.to_numeric(df.get("월").astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")

    # 숫자 파싱(괄호=음수, 콤마 제거) - 스케일 변환은 아직 X
    def _to_num(x):
        s = str(x).strip() if x is not None else ""
        if s == "" or s.lower() == "nan": return np.nan
        neg = s.startswith("(") and s.endswith(")")
        s = s.replace("(","").replace(")","").replace(",","")
        try:
            v = float(s)
            return -abs(v) if neg else v
        except:
            return np.nan

    df["실적"] = df["실적"].map(_to_num)

    # ── 2) 합계 시리즈 만들기(전월/당월/누적), 금액 지표만 /1e6 고정 스케일 ──
    def _sum_metric_fixed(y: int, m: int, tag: str) -> pd.Series:
        sub = df[(df["연도"]==y) & (df["월"]==m) & (df["구분4"]==tag)]
        if sub.empty: return pd.Series(dtype=float)
        s = sub.groupby("구분3")["실적"].sum()
        # 금액 지표만 백만원 스케일
        for k in money_metrics:
            if k in s.index and pd.notnull(s[k]):
                s[k] = float(s[k]) / 1_000_000.0
        return s

    def _sum_metric_cum_fixed(y: int, m: int, tag: str) -> pd.Series:
        sub = df[(df["연도"]==y) & (df["월"]<=m) & (df["구분4"]==tag)]
        if sub.empty: return pd.Series(dtype=float)
        s = sub.groupby("구분3")["실적"].sum()
        for k in money_metrics:
            if k in s.index and pd.notnull(s[k]):
                s[k] = float(s[k]) / 1_000_000.0
        return s

    # 전월 계산
    prev_y, prev_m = (year-1, 12) if month == 1 else (year, month-1)

    s_prev  = _sum_metric_fixed(prev_y, prev_m, "실적")
    s_cplan = _sum_metric_fixed(year,   month,  "계획")
    s_cact  = _sum_metric_fixed(year,   month,  "실적")
    s_yplan = _sum_metric_cum_fixed(year, month, "계획")
    s_yact  = _sum_metric_cum_fixed(year, month, "실적")

    rows = ['매출액','판매량','영업이익','%(영업)','순금융비용','경상이익','%(경상)']
    cols = ['전월','당월 계획','당월 실적','당월 계획대비','당월 전월대비','누적 계획','누적 실적','누적 계획대비']
    out = pd.DataFrame(index=rows, columns=cols, dtype=object)

    def g(sr, k):
        v = sr.get(k, np.nan)
        try: return float(v)
        except: return np.nan

    # ── 3) 금액/수량 행 포맷(정수/증감) ──
    for mkey in ['매출액','판매량','영업이익','순금융비용','경상이익']:
        prev = g(s_prev,  mkey)
        cp   = g(s_cplan, mkey)
        ca   = g(s_cact,  mkey)
        yp   = g(s_yplan, mkey)
        ya   = g(s_yact,  mkey)

        out.at[mkey,'전월']            = _fmt_int(prev)
        out.at[mkey,'당월 계획']        = _fmt_int(cp)
        out.at[mkey,'당월 실적']        = _fmt_int(ca)
        out.at[mkey,'당월 계획대비']     = _fmt_pp0(_diff(ca, cp))
        out.at[mkey,'당월 전월대비']     = _fmt_pp0(_diff(ca, prev))
        out.at[mkey,'누적 계획']         = _fmt_int(yp)
        out.at[mkey,'누적 실적']         = _fmt_int(ya)
        out.at[mkey,'누적 계획대비']      = _fmt_pp0(_diff(ya, yp))

    # ── 4) 퍼센트 행(이익/매출액*100) ──
    def _ratio_safe(n, d):
        try:
            n = float(n); d = float(d)
            if d == 0 or np.isnan(n) or np.isnan(d): return None
            return n/d*100
        except: return None

    op_prev  = _ratio_safe(g(s_prev,'영업이익'),  g(s_prev,'매출액'))
    op_plan  = _ratio_safe(g(s_cplan,'영업이익'), g(s_cplan,'매출액'))
    op_curr  = _ratio_safe(g(s_cact,'영업이익'),  g(s_cact,'매출액'))
    op_yplan = _ratio_safe(g(s_yplan,'영업이익'), g(s_yplan,'매출액'))
    op_yact  = _ratio_safe(g(s_yact,'영업이익'),  g(s_yact,'매출액'))

    or_prev  = _ratio_safe(g(s_prev,'경상이익'),  g(s_prev,'매출액'))
    or_plan  = _ratio_safe(g(s_cplan,'경상이익'), g(s_cplan,'매출액'))
    or_curr  = _ratio_safe(g(s_cact,'경상이익'),  g(s_cact,'매출액'))
    or_yplan = _ratio_safe(g(s_yplan,'경상이익'), g(s_yplan,'매출액'))
    or_yact  = _ratio_safe(g(s_yact,'경상이익'),  g(s_yact,'매출액'))

    # 영업이익률
    out.at['%(영업)','전월']          = _fmt_pct(op_prev)
    out.at['%(영업)','당월 계획']      = _fmt_pct(op_plan)
    out.at['%(영업)','당월 실적']      = _fmt_pct(op_curr)
    out.at['%(영업)','누적 계획']      = _fmt_pct(op_yplan)
    out.at['%(영업)','누적 실적']      = _fmt_pct(op_yact)
    out.at['%(영업)','당월 계획대비']   = _fmt_pp1(_diff(op_curr, op_plan))
    out.at['%(영업)','당월 전월대비']   = _fmt_pp1(_diff(op_curr, op_prev))

    # 경상이익률
    out.at['%(경상)','전월']          = _fmt_pct(or_prev)
    out.at['%(경상)','당월 계획']      = _fmt_pct(or_plan)
    out.at['%(경상)','당월 실적']      = _fmt_pct(or_curr)
    out.at['%(경상)','누적 계획']      = _fmt_pct(or_yplan)
    out.at['%(경상)','누적 실적']      = _fmt_pct(or_yact)
    out.at['%(경상)','당월 계획대비']   = _fmt_pp1(_diff(or_curr, or_plan))
    out.at['%(경상)','당월 전월대비']   = _fmt_pp1(_diff(or_curr, or_prev))

    # 메타
    out.attrs["units"] = {"money": "백만원(고정스케일)", "qty": "톤", "rate": "%"}
    out.attrs["used_month"] = int(month)
    out.attrs["prev_month"] = int(prev_m)
    return out




def create_item_pl_table_simple(
    year: int,
    month: int,
    data: pd.DataFrame,
    main_items: list[str] = ("CHQ", "CD", "STS", "BTB", "PB"),
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    df = create_pl_separate_hq_snapshot(data)

    sub = df[(df["연도"] == int(year)) & (df["월"] == int(month)) & (df["구분4"] == "매출액")].copy()
    cols = ["합계", *list(main_items), "상품 등"]

    rows = ["매출액","판매량","영업이익","%(영업)","경상이익","%(경상)"]
    out = pd.DataFrame(index=rows, columns=cols, dtype=float)

    if sub.empty:
        return out

    gi = sub.groupby(["구분1","구분3"])["실적"].sum()
    metrics_n = ["매출액","판매량","영업이익","경상이익"]

    for m in metrics_n:
        tot = gi.xs(m, level="구분3", drop_level=False).sum() if (m in gi.index.get_level_values(1)) else 0.0
        out.loc[m, "합계"] = float(tot)

        s_main = 0.0
        for it in main_items:
            v = gi.get((it, m), np.nan)
            try: v = float(v)
            except: v = np.nan
            out.loc[m, it] = v
            if pd.notnull(v): s_main += v

        out.loc[m, "상품 등"] = float(tot) - float(s_main)

    # 퍼센트(영업/경상) 계산
    den = out.loc["매출액"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out.loc["%(영업)"]  = np.where((den!=0)&np.isfinite(den)&np.isfinite(out.loc["영업이익"]),  (out.loc["영업이익"]/den)*100.0, np.nan)
        out.loc["%(경상)"]  = np.where((den!=0)&np.isfinite(den)&np.isfinite(out.loc["경상이익"]),  (out.loc["경상이익"]/den)*100.0, np.nan)

    return out


#품목손익 별도
import pandas as pd
import numpy as np
import re

def _parse_number(x) -> float:
    """천단위 콤마/괄호(음수)/공백을 안전하게 숫자로 변환."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    # (12,345) → -12345
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        # '1 234' 같은 경우 한 번 더 정리
        s = re.sub(r"\s+", "", s)
        try:
            return float(s)
        except Exception:
            return np.nan

def create_item_pl_from_flat(
    data: pd.DataFrame,
    year: int,
    month: int,
    main_items: tuple[str, ...] = ("CHQ","CD","STS","BTB","PB"),
    filter_tag: str = "품목손익"
) -> pd.DataFrame:
    """
    주어진 납작한 테이블(구분1~4, 연도, 월, 실적)을 품목손익 표로 변환.
    반환값: 숫자 DataFrame (포맷팅은 페이지에서)
    index: ['매출액','판매량','영업이익','%(영업)','경상이익','%(경상)']
    columns: ['합계', *main_items, '상품 등']
    """
    df = data.copy()

    # 문자열/숫자 정리
    for c in ["구분1","구분2","구분3","구분4"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str).str.strip()

    # 연/월 숫자화
    df["연도"] = pd.to_numeric(df.get("연도"), errors="coerce").astype("Int64")
    df["월"]   = pd.to_numeric(df.get("월"), errors="coerce").astype("Int64")

    # 실적 숫자화(콤마/괄호 대응)
    df["실적"] = df["실적"].apply(_parse_number)

    # 본사 + 해당 연월 + 품목손익 자료만
    mask = (
        df["구분2"].eq("본사") &
        df["연도"].eq(int(year)) &
        df["월"].eq(int(month)) &
        df["구분1"].str.contains(filter_tag)
    )
    sub = df.loc[mask, ["구분3","구분4","실적"]].copy()

    if sub.empty:
        cols = ["합계", *list(main_items), "상품 등"]
        rows = ["매출액","판매량","영업이익","%(영업)","경상이익","%(경상)"]
        return pd.DataFrame(index=rows, columns=cols, dtype=float)

    # 피벗: 행=지표(구분3), 열=품목/합계(구분4)
    pvt = sub.pivot_table(index="구분3", columns="구분4", values="실적", aggfunc="sum")

    # 출력 틀
    cols = ["합계", *list(main_items), "상품 등"]
    rows = ["매출액","판매량","영업이익","%(영업)","경상이익","%(경상)"]
    out  = pd.DataFrame(index=rows, columns=cols, dtype=float)

    # 숫자 지표 채우기
    money_metrics = ["매출액", "영업이익", "경상이익"]  # ← 백만원으로 변환할 지표

    for m in ["매출액","판매량","영업이익","경상이익"]:
        # 합계
        tot = pvt.get("합계", pd.Series(dtype=float)).get(m, np.nan)
        if pd.notnull(tot) and m in money_metrics:
            tot = float(tot) / 1_000_000.0
        out.at[m, "합계"] = float(tot) if pd.notnull(tot) else np.nan

        s_main = 0.0
        for it in main_items:
            v = pvt.get(it, pd.Series(dtype=float)).get(m, np.nan)
            if pd.notnull(v) and m in money_metrics:
                v = float(v) / 1_000_000.0
            else:
                v = float(v) if pd.notnull(v) else np.nan
            out.at[m, it] = v
            if pd.notnull(v):
                s_main += v

        # 상품 등 = 합계 − 지정 품목 합
        out.at[m, "상품 등"] = (out.at[m, "합계"] - s_main) if pd.notnull(out.at[m, "합계"]) else np.nan

        if m == "판매량":
            out.at[m, "상품 등"] = ""


    # 퍼센트 계산(매출액 분모)
    den = out.loc["매출액", cols].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        op = out.loc["영업이익", cols].astype(float)
        rp = out.loc["경상이익", cols].astype(float)
        out.loc["%(영업)", cols] = np.where((den!=0)&np.isfinite(den)&np.isfinite(op), (op/den)*100.0, np.nan)
        out.loc["%(경상)", cols] = np.where((den!=0)&np.isfinite(den)&np.isfinite(rp), (rp/den)*100.0, np.nan)
    
    out.at["%(영업)", "상품 등"] = ""
    out.at["%(경상)", "상품 등"] = ""

    return out

## 수정원가기준 손익 (별도)
import pandas as pd
import numpy as np
import re

def _parse_number(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if not s: return np.nan
    neg = s.startswith("(") and s.endswith(")")
    if neg: s = s[1:-1]
    s = s.replace(",", "").replace("%","")
    try:
        v = float(s)
        return -v if neg else v
    except:
        try:
            v = float(re.sub(r"\s+","",s))
            return -v if neg else v
        except:
            return np.nan

def create_item_change_cost_from_flat(
    data: pd.DataFrame,
    year: int,
    month: int,
    col_order=("계","CHQ","CD","STS","BTB","PB","내수","수출"),
):

    # ---- 정리 ----
    df = data.copy()
    for c in ["구분1","구분2","구분3","구분4"]:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].astype(str).str.replace("\n"," ").str.strip()

    df["연도"] = pd.to_numeric(df.get("연도"), errors="coerce").astype("Int64")
    df["월"]   = pd.to_numeric(df.get("월").astype(str).str.extract(r"(\d+)")[0],
                               errors="coerce").astype("Int64")
    if "실적" in df.columns:
        df["실적"] = df["실적"].apply(_parse_number)

    # 연월만 필터 (이 파일은 이미 ‘수정원가기준 별도’만 들어 있음)
    sub = df[(df["연도"].eq(int(year))) & (df["월"].eq(int(month)))].copy()

    # 피벗: index=지표(구분2), columns=구분3(계/CHQ/…)
    if sub.empty:
        present_cols = [c for c in col_order]  # 비어 있어도 프레임 반환
        rows = ["매출액","판매량","X등급 및 재고평가","영업이익","%(영업)","한계이익","%(한계)"]
        return pd.DataFrame(index=rows, columns=present_cols, dtype=object).fillna("")

    pvt = sub.pivot_table(index="구분2", columns="구분3", values="실적", aggfunc="last")
    pvt.index = [s.replace("  "," ").strip() for s in pvt.index]  # 라벨 정리

    # 실제 존재하는 열만 사용 (데이터에 없으면 자동 제외)
    present_cols = [c for c in col_order if c in pvt.columns]
    rows = ["매출액","판매량","X등급 및 재고평가","영업이익","%(영업)","한계이익","%(한계)"]
    out = pd.DataFrame(index=rows, columns=present_cols, dtype=float)

    def gv(metric, col):
        try: return pvt.loc[metric, col]
        except KeyError: return np.nan

    # 1) 기본 지표 채우기
    for r in ["매출액","판매량","영업이익","X등급 및 재고평가"]:
        for c in present_cols:
            out.at[r, c] = gv(r, c)

    # 2) 파생: 한계이익
    for c in present_cols:
        e = out.at["영업이익", c]
        x = out.at["X등급 및 재고평가", c]
        out.at["한계이익", c] = (0 if pd.isna(e) else float(e)) + (0 if pd.isna(x) else float(x))
        if pd.isna(e) and pd.isna(x):
            out.at["한계이익", c] = np.nan

    # 3) 퍼센트 계산 (파일의 % 행은 사용하지 않고 직접 산출)
    for pct_name, num_name in [("%(영업)","영업이익"),("%(한계)","한계이익")]:
        den = out.loc["매출액", present_cols].astype(float)
        num = out.loc[num_name, present_cols].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.loc[pct_name, present_cols] = np.where(
                (den!=0)&np.isfinite(den)&np.isfinite(num), (num/den)*100.0, np.nan
            )

    # 4) NaN → 공란
    out = out.astype("object")
    out.iloc[:, :] = out.where(pd.notnull(out), "")

    return out


## 제품수불표

# === 제품수불표: 연산 전용 ===
import pandas as pd
import numpy as np
import re

def _pf_to_num(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace(",", "", regex=False).str.replace(r"\s+", "", regex=True)
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(0.0)

def _clean_product_flow(df_raw: pd.DataFrame) -> pd.DataFrame:
    need = {"구분1","구분2","구분3","연도","월","실적"}
    miss = need - set(df_raw.columns)
    if miss:
        raise ValueError(f"[제품수불표] 필수 컬럼 누락: {miss}")
    df = df_raw.copy()

    # 텍스트 정규화
    for c in ["구분1","구분2","구분3","구분4"]:
        if c not in df.columns: df[c] = ""
        df[c] = (df[c].astype(str)
                     .str.replace("\xa0", " ")
                     .str.replace(r"\s+", " ", regex=True)
                     .str.strip())

    # 숫자화
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df["월"]   = pd.to_numeric(df["월"],   errors="coerce").astype("Int64")
    df["실적"] = _pf_to_num(df["실적"])

    # 제품수불표만
    df = df[df["구분1"] == "제품수불표"].copy()
    # 동일(구분2,구분3)이 여러 줄이면 ‘마지막 값’ 사용하도록 순서 기억
    df["__ord__"] = range(len(df))
    return df

def create_product_flow_base(year: int,
                             month: int,
                             data: pd.DataFrame,
                             amount_div: float = 1_000_000) -> pd.DataFrame:
    """
    원천 데이터에서 해당 연·월의 값을 뽑아 1행짜리 '연산 결과 테이블'을 돌려준다.
    - 반환 컬럼(숫자만, 포맷/헤더 없음):
      ['입고-기초_단가','입고-기초_금액','매출원가-기초_단가','매출원가-기초_금액']
    - 금액은 amount_div로 나눠 단위를 맞춘다(기본: 원→백만원).
    """
    df = _clean_product_flow(data)

    sel = df[(df["연도"] == year) & (df["월"] == month)].copy()
    # 마지막 값 우선
    sel = sel.sort_values("__ord__").drop_duplicates(["구분2","구분3"], keep="last")

    # 피벗(없으면 0.0)
    pv = sel.pivot_table(index="구분2", columns="구분3", values="실적", aggfunc="last")
    get = lambda g2, g3: float(pv.get(g3, pd.Series()).get(g2, 0.0))

    in_unit     = get("입고-기초", "단가")
    in_amount   = get("입고-기초", "금액") / amount_div
    cogs_unit   = get("매출원가-기초", "단가")
    cogs_amount = get("매출원가-기초", "금액") / amount_div

    base = pd.DataFrame([{
        "입고-기초_단가": in_unit,
        "입고-기초_금액": in_amount,
        "매출원가-기초_단가": cogs_unit,
        "매출원가-기초_금액": cogs_amount,
    }])
    # 나중에 화면에서 쓸 메타 정보도 같이 보관(옵션)
    base.attrs["unit_label"] = "백만원" if amount_div == 1_000_000 else ""
    base.attrs["year"] = year
    base.attrs["month"] = month
    return base


## 현금흐름표 별도

import pandas as pd

def _to_num_cf(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.replace(",", "", regex=False).str.strip()
    v = pd.to_numeric(s, errors="coerce")
    return v.fillna(0.0)

def _clean_cf_separate(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    need = {"구분1","구분2","연도","월","실적"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"필수 컬럼 누락: {miss}")
    for c in ["구분1","구분2","구분3","구분4"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df["월"]   = pd.to_numeric(df["월"],   errors="coerce").astype("Int64")
    df["실적"] = _to_num_cf(df["실적"])
    # 별도만 사용
    df = df[df["구분1"] == "현금흐름표_별도"].copy()
    # 원본 등장 순서 보존용
    df["__ord__"] = range(len(df))
    return df

def create_cashflow_separate_by_order(year: int, month: int, data: pd.DataFrame, item_order: list[str]) -> pd.DataFrame:
    """
    - 행: item_order 순서를 그대로 사용 (중복 라벨 허용; 예: '기타' 2번)
    - 열: (year-2)년, (year-1)년, 전월누적, 당월누적, (year)년누적
    - 중복 라벨은 각 (연,월) 그룹에서 '그 라벨의 n번째 등장'만 뽑아 누적 합산
    """
    df = _clean_cf_separate(data)

    # 사용 월 폴백(요청 연도 기준)
    avail = sorted(df.loc[df["연도"] == year, "월"].dropna().unique())
    used_month = int(month)
    if len(avail) and month not in avail:
        past = [m for m in avail if m <= month]
        used_month = int(max(past) if past else max(avail))

    # item_order에서 같은 라벨(예: '기타')의 n번째 등장 번호 지정
    name_counts = {}
    order_with_n = []
    for name in item_order:
        name_counts[name] = name_counts.get(name, 0) + 1
        order_with_n.append((name, name_counts[name]))  # ('기타', 1), ('기타', 2) ...

    # (핵심) 라벨의 n번째 등장만 골라 합산
    def _sum_item_nth(name: str, nth: int, years, months):
        sub = df[(df["연도"].isin(years)) & (df["월"].isin(months))]
        total = 0.0
        # (연,월)별로, 해당 라벨의 nth 등장 행만 집계
        for (_, _), g in sub.groupby(["연도", "월"], sort=False):
            g = g[g["구분2"] == name].sort_values("__ord__", kind="stable")
            if len(g) >= nth:
                total += float(g.iloc[nth - 1]["실적"])
        return total

    def _block(years, months):
        return [_sum_item_nth(nm, nth, years, months) for (nm, nth) in order_with_n]

    col_prev2_label = f"{str(year-2)[-2:]}년"
    col_prev1_label = f"{str(year-1)[-2:]}년"
    col_currsum_label = f"{str(year)[-2:]}년누적"

    col_prev2 = _block([year-2], range(1, 13))
    col_prev1 = _block([year-1], range(1, 13))
    prev_months = range(1, used_month) if used_month > 1 else []
    col_prev   = _block([year], prev_months) if prev_months else [0.0] * len(order_with_n)
    col_ytd    = _block([year], range(1, used_month + 1))

    # 인덱스는 원래 라벨 그대로
    index_labels = [nm for (nm, _) in order_with_n]

    out = pd.DataFrame(
        {
            col_prev2_label: col_prev2,
            col_prev1_label: col_prev1,
            "전월누적": col_prev,
            "당월누적": col_ytd,
            col_currsum_label: col_ytd,
        },
        index=pd.Index(index_labels, name="구분"),
        dtype=float
    )


    out.attrs["used_month"] = used_month
    out.attrs["prev_month"] = (used_month - 1) if used_month > 1 else 1
    return out


#####재무상태표 별도

import pandas as pd
import numpy as np

def _bs_to_number(x):
    s = str(x).strip()
    if not s:
        return 0.0
    neg = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace(',', '')
    try:
        v = float(s)
    except Exception:
        return 0.0
    return -abs(v) if neg else v


import pandas as pd
import numpy as np

def _bs_to_number(x):
    s = str(x).strip()
    if not s:
        return 0.0
    neg = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace(',', '')
    try:
        v = float(s)
    except Exception:
        return 0.0
    return -abs(v) if neg else v

def _normalize_bs_simple(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    item_col = next((c for c in ['구분3','구분2','항목','구분'] if c in df.columns), None)
    comp_col = next((c for c in ['회사','법인','구분4'] if c in df.columns), None)

    for need in ['연도','월','실적']:
        if need not in df.columns:
            raise ValueError(f"재무상태표 데이터에 '{need}' 컬럼이 필요합니다.")
    if item_col is None:
        raise ValueError("재무상태표: '구분3/구분2/항목/구분' 중 하나가 필요합니다.")

    df[item_col] = df[item_col].astype(str).str.strip()
    if comp_col is None:
        comp_col = '_회사'
        df[comp_col] = '전체'
    else:
        df[comp_col] = df[comp_col].astype(str).str.strip().replace({'타이':'태국'})

    s = df['실적'].astype(str).str.strip()
    neg = s.str.match(r'^\(.*\)$')
    s = s.str.replace(r'[(),]', '', regex=True)
    v = pd.to_numeric(s, errors='coerce').fillna(0.0)
    v[neg] = -v[neg].abs()
    df['실적'] = v

    df['연도'] = pd.to_numeric(df['연도'], errors='coerce').astype('Int64')
    df['월']   = pd.to_numeric(df['월'],   errors='coerce').astype('Int64')

    drop_cols = [c for c in df.columns if str(c).startswith('Unnamed')]
    if drop_cols: df = df.drop(columns=drop_cols, errors='ignore')

    return df.rename(columns={item_col:'구분3', comp_col:'회사'})


def _pick_used_prev_month(df: pd.DataFrame, year: int, month: int):
    # year, month에서 가장 가까운 사용월
    used = _closest_month(df, year, month)
    if used is None:
        # 그 해에 있는 최대 월로 대체
        y_months = df.loc[df['연도']==year, '월'].dropna().astype(int)
        used = int(y_months.max()) if not y_months.empty else None
    if used is None:
        # 전년도 12월로 최종 대체
        used = _closest_month(df, year-1, 12)

    if used is None:
        raise ValueError(f"{year}년에 사용할 월 데이터가 없습니다.")

    # 전월 계산(해당 해에서 직전월 없으면 전년도 12월)
    if used > 1:
        prev_y = year
        prev_m = _closest_month(df, year, used-1)
        if prev_m is None:
            prev_m = used-1
    else:
        prev_y = year-1
        prev_m = _closest_month(df, year-1, 12)

    return int(used), int(prev_y) if prev_y is not None else None, int(prev_m) if prev_m is not None else None


import pandas as pd

import pandas as pd

def create_bs_from_gubun2_teuksugang(
    year: int,
    month: int,
    data: pd.DataFrame,
    item_order: list[str],
) -> pd.DataFrame:
    """
    원본 데이터의 '구분2'에서 '특수강'만 필터링 후 재무상태표 생성.
    반환 컬럼: ['\'YY년말', '\'YY년 선택전 월', '\'YY년 선택월', '전월대비']
    """
    # 1) 원본에서 '구분2'로 선필터 (정규화 이전에 '구분2'만 신뢰)
    if '구분2' not in data.columns:
        raise ValueError("'구분2' 컬럼이 없습니다. 원본 스키마를 확인하세요.")
    df_src = data.copy()
    df_src['구분2'] = df_src['구분2'].astype(str).str.strip()
    df_src = df_src[df_src['구분2'] == '특수강'].copy()

    if df_src.empty:
        raise ValueError("원본 '구분2'에서 '특수강' 데이터를 찾지 못했습니다.")

    # 2) 정규화(숫자 파싱/타입 캐스팅 등)
    df = _normalize_bs_simple(df_src)  # 이 함수는 열 이름을 '구분3','연도','월','실적'로 정돈

    # 3) 해당 연도의 사용월(선택월) 결정
    y_months = sorted(int(m) for m in df.loc[df['연도']==year, '월'].dropna().unique())
    if not y_months:
        raise ValueError(f"'구분2=특수강'의 {year}년 데이터가 없습니다.")

    req_m = int(month)
    used_candidates = [m for m in y_months if m <= req_m]
    used_m = used_candidates[-1] if used_candidates else y_months[-1]  # 같은 해에서 가장 근접 과거월, 없으면 그 해 최대월

    # 4) 선택전 월 결정 (같은 해 직전 존재 월 → 없으면 전년도 12월 → 그래도 없으면 None=0 처리)
    prev_y = year
    prev_list = [m for m in y_months if m < used_m]
    if prev_list:
        prev_m = prev_list[-1]
    else:
        prev_y = year - 1
        prev_m = 12 if not df[(df['연도']==prev_y) & (df['월']==12)].empty else None

    # 5) 합계 헬퍼
    def _sum_item(item: str, y: int | None, m: int | None) -> float:
        if y is None or m is None:
            return 0.0
        mask = (df['구분3'].astype(str).str.strip()==item) & (df['연도']==y) & (df['월']==m)
        return float(df.loc[mask, '실적'].sum())

    # 6) 컬럼 라벨
    yy_prev = f"{(year-1)%100:02d}"
    yy_curr = f"{year%100:02d}"
    col_yend = f"'{yy_prev}년말"
    col_prev = f"'{yy_curr}년 선택전 월"
    col_curr = f"'{yy_curr}년 선택월"

    # 7) 행 구성 (item_order 그대로)
    rows = []
    for item in item_order:
        v_yend = _sum_item(item, year-1, 12)       # 전년도 12월만
        v_prev = _sum_item(item, prev_y, prev_m)   # 선택전 월(없으면 0)
        v_curr = _sum_item(item, year, used_m)     # 선택월
        rows.append({col_yend: v_yend, col_prev: v_prev, col_curr: v_curr, "전월대비": v_curr - v_prev})

    out = pd.DataFrame(rows, index=pd.Index(item_order, name='구분')).fillna(0.0)
    # ... out 생성까지 동일 ...

    # (추가) 뷰 호환용 컬럼명으로 리네임
    out = out.rename(columns={
        col_yend: "'24년말",         # 전년도 12월
        col_prev: "'25",         # 선택전 월(전월)
        col_curr: "당월",         # 선택월
        "전월대비": "전월비 증감"
    })

    # (고정 순서)
    out = out[["'24년말","'25","당월","전월비 증감"]]


    # 8) 뷰용 메타 (항상 int/또는 None)
    out.attrs['used_month'] = int(used_m)
    out.attrs['prev_month'] = int(prev_m) if prev_m is not None else None
    out.attrs['prev_year']  = int(prev_y)

    return out




    # 메타
    out.attrs['used_month'] = used_m
    out.attrs['prev_month'] = prev_m
    return out



###회전일 (별도)

def _normalize_turnover_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 값/항목/회사 후보
    val_col = next((c for c in ['회전일','실적','값','value'] if c in df.columns), None)
    if val_col is None:
        raise ValueError("회전일 데이터에 '회전일/실적/값' 컬럼이 필요합니다.")

    # 기본 숫자/연월 정리
    df['연도'] = pd.to_numeric(df['연도'], errors='coerce').astype('Int64')
    df['월']   = pd.to_numeric(df['월'],   errors='coerce').astype('Int64')

    # (1) 괄호 음수 → 음수, 쉼표 제거
    df[val_col] = _num_paren(df[val_col]).astype(float)

    # (2) **구분3 == '특수강'만 필터(별칭 정리 포함)**  ← 추가
    if '구분3' in df.columns:
        g3norm = (df['구분3'].astype(str).str.strip()
                            .replace({'타이':'태국', '남동':'남통'}))
        df = df[g3norm.eq('특수강')].copy()
        df.loc[:, '구분3'] = g3norm.loc[df.index]  # 정리된 라벨로 유지

    # 후보 열들
    cand_item = ['구분3','구분2','구분','항목']
    cand_comp = ['회사','법인','구분4']

    # 회사/항목 기본 선택
    item_col = next((c for c in cand_item if c in df.columns), None)
    comp_col = next((c for c in cand_comp if c in df.columns), None)

    # --- 회사명이 구분3에 들어있는 케이스 감지 ---
    known_companies = {'계','전체','연결','특수강','본사','남통','남동','천진','태국','타이'}
    def looks_like_company(series) -> bool:
        s = series.astype(str).str.strip()
        nonnull = s[s != ""]
        if len(nonnull) == 0: return False
        hit = nonnull.isin(known_companies).mean()
        return hit >= 0.7

    if ('구분3' in df.columns) and (comp_col is None or looks_like_company(df['구분3'])):
        comp_col = '구분3'
        item_col = next((c for c in ['구분2','구분','항목'] if c in df.columns and c != comp_col), item_col)

    if item_col is None:
        raise ValueError("회전일 데이터에 항목 컬럼(구분3/구분2/구분/항목) 중 하나가 필요합니다.")

    # 문자열 정리 + alias
    df[item_col] = df[item_col].astype(str).str.strip()
    if comp_col is None:
        comp_col = '_회사'
        df[comp_col] = '전체'
    df[comp_col] = (
        df[comp_col].astype(str).str.strip()
          .replace({'타이':'태국', '남동':'남통'})
    )

    # (3) **최종 회사 라벨을 ‘특수강’으로 강제**  ← 추가
    # 구분3을 회사로 쓰지 않는 입력 형태라도, 계산 결과는 특수강만 되게 고정
    if comp_col != '구분3':
        df[comp_col] = '특수강'

    # 쓰지 않는 Unnamed 열 제거
    drop_cols = [c for c in df.columns if str(c).startswith('Unnamed')]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    return df.rename(columns={item_col: '구분', comp_col: '회사', val_col: '값'})




import numpy as np
import pandas as pd

# modules.py

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 공통 유틸: 괄호음수 + 쉼표 제거
# ─────────────────────────────────────────────────────────────
def _num_paren(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    neg = s.str.match(r'^\(.*\)$')
    s = s.str.replace(r'[(),]', '', regex=True)
    v = pd.to_numeric(s, errors='coerce')
    v[neg] = -v[neg].abs()
    return v


# ─────────────────────────────────────────────────────────────
# 원천 정규화 (표준 스키마: 구분 / 회사 / 값 / 연도 / 월)
# ─────────────────────────────────────────────────────────────
def _normalize_turnover_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    회전일 원천을 표준 스키마로 정리:
      - 항목: '구분'
      - 회사: '회사'
      - 값  : '값'
    회사명이 구분3(계/특수강/남통/천진/태국 등)에 들어있는 경우도 자동 감지.
    """
    df = df_raw.copy()

    # 값/항목/회사 후보
    val_col = next((c for c in ['회전일','실적','값','value'] if c in df.columns), None)
    if val_col is None:
        raise ValueError("회전일 데이터에 '회전일/실적/값' 컬럼이 필요합니다.")

    # 기본 숫자/연월 정리
    df['연도'] = pd.to_numeric(df['연도'], errors='coerce').astype('Int64')
    df['월']   = pd.to_numeric(df['월'],   errors='coerce').astype('Int64')

    # 괄호 음수 → 음수, 쉼표 제거
    df[val_col] = _num_paren(df[val_col]).astype(float)

    # 후보 열들
    cand_item = ['구분3','구분2','구분','항목']
    cand_comp = ['회사','법인','구분4']

    # 회사/항목 기본 선택
    item_col = next((c for c in cand_item if c in df.columns), None)
    comp_col = next((c for c in cand_comp if c in df.columns), None)

    # 회사명이 구분3에 들어있는 케이스 감지
    known_companies = {'계','전체','연결','특수강','본사','남통','남동','천진','태국','타이'}
    def looks_like_company(series) -> bool:
        s = series.astype(str).str.strip()
        nonnull = s[s != ""]
        if len(nonnull) == 0: return False
        hit = nonnull.isin(known_companies).mean()
        return hit >= 0.7

    if ('구분3' in df.columns) and (comp_col is None or looks_like_company(df['구분3'])):
        comp_col = '구분3'
        # 항목은 구분2 > 구분 > 항목 순으로
        item_col = next((c for c in ['구분2','구분','항목'] if c in df.columns and c != comp_col), item_col)

    if item_col is None:
        raise ValueError("회전일 데이터에 항목 컬럼(구분3/구분2/구분/항목) 중 하나가 필요합니다.")

    # 문자열 정리 + alias
    df[item_col] = df[item_col].astype(str).str.strip()
    if comp_col is None:
        comp_col = '_회사'
        df[comp_col] = '전체'
    df[comp_col] = (
        df[comp_col].astype(str).str.strip()
          .replace({'타이':'태국', '남동':'남통'})   # alias 통일
    )

    # 쓰지 않는 Unnamed 열 제거
    drop_cols = [c for c in df.columns if str(c).startswith('Unnamed')]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    return df.rename(columns={item_col: '구분', comp_col: '회사', val_col: '값'})


# ─────────────────────────────────────────────────────────────
# 특수강 전용 4열 표 생성
#  - '전년도 말' 컬럼은 (year, 1월) 데이터를 사용
#  - 구분3 == '특수강' 행만 사용해 연산
#  - 계산 단계에서 반올림/포맷 없음 (정밀도 보존)
# ─────────────────────────────────────────────────────────────
def create_turnover_special_steel(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:
    """
    행: 매출채권 / 재고자산 / 매임채무 / 현금전환주기
    열: '{YY}년말'(실데이터: year의 1월) / '{YY}.{m-1} 월' / '{YY}.{m} 월' / '전월대비'
    - 동일 항목·월 중복행은 평균 사용
    - 구분3 == '특수강' 조건으로 원천 제한
    - 계산 결과는 반올림하지 않음(표시단에서 포맷)
    """
    # 1) 원천 정규화
    df_norm = _normalize_turnover_v2(data)

    # alias 통일 함수
    def norm_comp(x: str) -> str:
        x = str(x).strip()
        x = {'타이': '태국', '남동': '남통'}.get(x, x)
        if x in ('전체', '연결'):
            return '계'
        return x

    # 2) 구분3 == '특수강'으로 필터 (원천 data 기준 → 인덱스 정합 유지)
    if '구분3' in data.columns:
        g3norm = (data['구분3'].astype(str).str.strip()
                               .replace({'타이':'태국', '남동':'남통'}))
        mask = g3norm.eq('특수강')
        df = df_norm.loc[mask].copy()
    else:
        # 구분3 없으면 정규화된 회사로 보조
        df = df_norm.loc[df_norm['회사'].map(norm_comp).eq('특수강')].copy()

    # 3) 당월 선택 (요청월이 없으면 가장 가까운 과거월)
    avail_this = sorted(df.loc[df['연도'] == year, '월'].dropna().unique())
    used_m = month
    if avail_this and month not in avail_this:
        prior = [m for m in avail_this if m <= month]
        used_m = prior[-1] if prior else avail_this[-1]

    # 4) 전월 계산
    prev_y, prev_m = year, used_m - 1
    if prev_m < 1:
        prev_y = year - 1
        prev_avail = sorted(df.loc[df['연도'] == prev_y, '월'].dropna().unique())
        prev_m = prev_avail[-1] if prev_avail else 12

    # 5) '전년도 말' → (year, 1월) 데이터 사용 (1월 없으면 해당 연도 가장 이른 월)
    yend_label_year = year - 1          # 라벨 표기용(예: '24년말)
    yend_data_year  = year              # 실제 참조 연도
    avail_y = sorted(df.loc[df['연도'] == yend_data_year, '월'].dropna().unique())
    if len(avail_y) == 0:
        yend_m = 1                      # 데이터가 전혀 없으면 1월로 가정(결국 NaN 반환)
    else:
        yend_m = 1 if 1 in avail_y else avail_y[0]

    # 6) 단일 월·항목 값 추출 (중복행은 평균)
    def val_of(item: str, y: int, m: int) -> float:
        sub = df[(df['구분'] == item) & (df['연도'] == y) & (df['월'] == m)].copy()
        if sub.empty:
            return np.nan
        sub['회사N'] = sub['회사'].map(norm_comp)
        byc = sub.groupby('회사N', dropna=True)['값'].mean()
        # 특수강 우선
        if '특수강' in byc.index:
            return float(byc['특수강'])
        # 특수강이 비어있으면 '계'라도 쓰고 싶으면 아래 주석 해제
        # elif '계' in byc.index:
        #     return float(byc['계'])
        return np.nan

    items = ['매출채권', '재고자산', '매임채무']

    yend = {it: val_of(it, yend_data_year, yend_m) for it in items}
    prev = {it: val_of(it, prev_y,           prev_m) for it in items}
    curr = {it: val_of(it, year,             used_m) for it in items}

    # 7) 현금전환주기 = 매출채권 + 재고자산 - 매임채무 (NaN-safe)
    def ccc(dic):
        a = dic.get('매출채권', np.nan)
        b = dic.get('재고자산', np.nan)
        c = dic.get('매임채무', np.nan)
        a = 0 if pd.isna(a) else a
        b = 0 if pd.isna(b) else b
        c = 0 if pd.isna(c) else c
        return a + b - c

    yend_ccc = ccc(yend)
    prev_ccc = ccc(prev)
    curr_ccc = ccc(curr)

    # 8) 표 구성 (반올림/포맷 없음)
    col_yend = f"'{str(yend_label_year)[-2:]}년말"     # 예: '24년말
    col_prev = f"'{str(prev_y)[-2:]}.{prev_m} 월"
    col_curr = f"'{str(year)[-2:]}.{used_m} 월"

    out = pd.DataFrame(
        index=['매출채권', '재고자산', '매임채무', '현금전환주기'],
        columns=[col_yend, col_prev, col_curr, '전월대비'],
        dtype=float
    )

    for it in items:
        out.loc[it, col_yend] = yend[it]
        out.loc[it, col_prev] = prev[it]
        out.loc[it, col_curr] = curr[it]
        out.loc[it, '전월대비'] = (
            curr[it] - prev[it] if pd.notnull(curr[it]) and pd.notnull(prev[it]) else np.nan
        )

    out.loc['현금전환주기', col_yend] = yend_ccc
    out.loc['현금전환주기', col_prev] = prev_ccc
    out.loc['현금전환주기', col_curr] = curr_ccc
    out.loc['현금전환주기', '전월대비'] = (
        curr_ccc - prev_ccc if pd.notnull(curr_ccc) and pd.notnull(prev_ccc) else np.nan
    )

    # 메타(뷰에서 캡션 등으로 사용 가능)
    out.attrs.update({
        'company': '특수강',
        'used_month': used_m,
        'prev_month': prev_m,
        'year_end_label_year': yend_label_year,   # '24년말 표기용
        'year_end_data': (yend_data_year, yend_m) # (실데이터 참조 연,월) = (year, 1 또는 earliest)
    })
    return out
