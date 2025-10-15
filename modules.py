import re
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st


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


def create_defect_summary_pohang(year:int,
                                 month:int,
                                 data:pd.DataFrame,
                                 months_window:tuple=(5,6,7),
                                 plant_name:str="포항") -> pd.DataFrame:
    """
    3-레벨 멀티인덱스 버전
    index (상위, 중위, 하위):
      ('',  '',   '공정성')  # CHQ-공정성
      ('',  '',   '소재성')  # CHQ-소재성
      ('',  'CHQ','')        # CHQ 합계
      ('',  ' ',  '공정성')  # CD-공정성  ← 중위 레벨을 ' ' (스페이스)로 둬 CHQ 블록과 구분
      ('',  ' ',  '소재성')  # CD-소재성
      ('',  'CD', '')        # CD 합계
      ('',  '공정성','')     # 전체 공정성
      ('',  '소재성','')     # 전체 소재성
      ('포항','','')         # 포항 총계
    """
    df = data.copy()

    # 형 변환/정규화
    for c in ['연도','월','실적']:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')
    for c in ['구분1','구분2','구분3','구분4']:
        if c not in df.columns: df[c] = ''
        df[c] = df[c].fillna('').astype(str)

    # 대상 공장 필터
    df = df[df['구분1'].str.contains(plant_name)]
    prev_year = year - 1
    mlist = list(months_window)

    # 안전 합/평균
    safe_sum  = lambda s: float(np.nansum(s))  if len(s) else 0.0
    safe_mean = lambda s: float(np.nanmean(s)) if len(s) else 0.0

    # 집계 헬퍼
    def pick(g2=None, g3=None, yy=None, mm=None, only_target=False):
        q = df.copy()
        if only_target:
            q = q[q['구분4'] == '목표']  # 목표가 이렇게 오면 사용
        if g2 is not None: q = q[q['구분2'] == g2]
        if g3 is not None: q = q[q['구분3'] == g3]
        if yy is not None: q = q[q['연도'] == yy]
        if mm is not None: q = q[q['월'] == mm]
        return safe_sum(q['실적'])

    # ===== 인덱스/컬럼 정의 =====
    rows = [
        ('','',  '공정성'),  ('','',  '소재성'),  ('','CHQ',''),
        ('',' ', '공정성'),  ('',' ', '소재성'),  ('','CD',''),
        ('','공정성',''),   ('','소재성',''),    ('포항','','')
    ]
    index = pd.MultiIndex.from_tuples(rows, names=['상','중','하'])
    

    month_cols = [f"{m}월" for m in mlist]
    col_prev_avg = f"{str(prev_year)[-2:]}년 월평균"
    col_target   = f"{str(year)[-2:]}년 목표"
    cols = [col_prev_avg, col_target] + month_cols + ['합계','월평균']

    out = pd.DataFrame(0.0, index=index, columns=cols)

    # ---------- ① 전년 월평균 ----------
    # CHQ
    chq_prev_ps = [pick(g2='CHQ', g3='공정성', yy=prev_year, mm=m) for m in range(1,13)]
    chq_prev_ms = [pick(g2='CHQ', g3='소재성', yy=prev_year, mm=m) for m in range(1,13)]
    out.loc[('', '', '공정성'), col_prev_avg] = safe_mean(chq_prev_ps)
    out.loc[('', '', '소재성'), col_prev_avg] = safe_mean(chq_prev_ms)
    out.loc[('', 'CHQ', ''),   col_prev_avg] = safe_mean([a+b for a,b in zip(chq_prev_ps, chq_prev_ms)])

    # CD  (중위 레벨 ' ' 블록)
    cd_prev_ps = [pick(g2='CD', g3='공정성', yy=prev_year, mm=m) for m in range(1,13)]
    cd_prev_ms = [pick(g2='CD', g3='소재성', yy=prev_year, mm=m) for m in range(1,13)]
    out.loc[('', ' ', '공정성'), col_prev_avg] = safe_mean(cd_prev_ps)
    out.loc[('', ' ', '소재성'), col_prev_avg] = safe_mean(cd_prev_ms)
    out.loc[('', 'CD', ''),      col_prev_avg] = safe_mean([a+b for a,b in zip(cd_prev_ps, cd_prev_ms)])

    # 전체/포항
    ps_all_prev = [pick(g3='공정성', yy=prev_year, mm=m) for m in range(1,13)]
    ms_all_prev = [pick(g3='소재성', yy=prev_year, mm=m) for m in range(1,13)]
    out.loc[('', '공정성', ''), col_prev_avg] = safe_mean(ps_all_prev)
    out.loc[('', '소재성', ''), col_prev_avg] = safe_mean(ms_all_prev)
    out.loc[('포항', '', ''),    col_prev_avg] = safe_mean([ps_all_prev[i]+ms_all_prev[i] for i in range(12)])

    # ---------- ② 당년 목표 (없으면 0) ----------
    out.loc[:, col_target] = 0.0
    # 필요하면 예: out.loc[('', '', '공정성'), col_target] = pick(g2='CHQ', g3='공정성', only_target=True)

    # ---------- ③ 선택월/합계/월평균 ----------
    # CHQ
    chq_ps = [pick(g2='CHQ', g3='공정성', yy=year, mm=m) for m in mlist]
    chq_ms = [pick(g2='CHQ', g3='소재성', yy=year, mm=m) for m in mlist]
    out.loc[('', '', '공정성'), month_cols] = chq_ps
    out.loc[('', '', '공정성'), ['합계','월평균']] = [safe_sum(chq_ps), safe_mean(chq_ps)]
    out.loc[('', '', '소재성'), month_cols] = chq_ms
    out.loc[('', '', '소재성'), ['합계','월평균']] = [safe_sum(chq_ms), safe_mean(chq_ms)]
    out.loc[('', 'CHQ',''), month_cols] = [chq_ps[i]+chq_ms[i] for i in range(len(mlist))]
    out.loc[('', 'CHQ',''), ['합계','월평균']] = [
        safe_sum(out.loc[('', 'CHQ',''), month_cols]),
        safe_mean(out.loc[('', 'CHQ',''), month_cols]),
    ]

    # CD
    cd_ps = [pick(g2='CD', g3='공정성', yy=year, mm=m) for m in mlist]
    cd_ms = [pick(g2='CD', g3='소재성', yy=year, mm=m) for m in mlist]
    out.loc[('', ' ', '공정성'), month_cols] = cd_ps
    out.loc[('', ' ', '공정성'), ['합계','월평균']] = [safe_sum(cd_ps), safe_mean(cd_ps)]
    out.loc[('', ' ', '소재성'), month_cols] = cd_ms
    out.loc[('', ' ', '소재성'), ['합계','월평균']] = [safe_sum(cd_ms), safe_mean(cd_ms)]
    out.loc[('', 'CD',''), month_cols] = [cd_ps[i]+cd_ms[i] for i in range(len(mlist))]
    out.loc[('', 'CD',''), ['합계','월평균']] = [
        safe_sum(out.loc[('', 'CD',''), month_cols]),
        safe_mean(out.loc[('', 'CD',''), month_cols]),
    ]

    # 전체 공정성/소재성 + 포항 총계
    ps_all = [pick(g3='공정성', yy=year, mm=m) for m in mlist]
    ms_all = [pick(g3='소재성', yy=year, mm=m) for m in mlist]
    out.loc[('', '공정성',''), month_cols] = ps_all
    out.loc[('', '공정성',''), ['합계','월평균']] = [safe_sum(ps_all), safe_mean(ps_all)]
    out.loc[('', '소재성',''), month_cols] = ms_all
    out.loc[('', '소재성',''), ['합계','월평균']] = [safe_sum(ms_all), safe_mean(ms_all)]
    total = [ps_all[i] + ms_all[i] for i in range(len(mlist))]
    out.loc[('포항','',''), month_cols] = total
    out.loc[('포항','',''), ['합계','월평균']] = [safe_sum(total), safe_mean(total)]

    # 반올림
    return out.round(0)



def create_defect_summary_chungju(
    year:int,
    month:int,
    data:pd.DataFrame,
    months_window:tuple,
    plant1_name:str="충주",      # 충주1공장
    plant2_name:str="충주2"      # 충주2공장 (CD만)
) -> pd.DataFrame:
    """
    행(9행, 계단 구조):
      0 ('',   '',  '공정성')      ┐  충주1 CHQ
      1 ('',   '',  '소재성')      ┘
      2 ('',  '충주1공장(CHQ)', '')  ← 충주1 합계(라벨행)

      3 ('',   '',  '공정성')      ┐  충주2 CD(마봉강)만 표기
      4 ('',   '',  '소재성')      ┘
      5 ('',   '충주2공장',      '')  ← 충주2 합계(라벨행)

      6 ('공정성','',  '')          전체 공정성
      7 ('소재성','',  '')          전체 소재성
      8 ('충주','',    '')          충주 총계
    """
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

    month_cols   = [f"{m}월" for m in mlist]
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
