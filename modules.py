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
    end_date = f"{year}-{month+1}"
    date_index = pd.date_range(end=end_date, periods=12, freq='M')
    return [f"{date.year % 100}년 {date.month}월" for date in date_index]

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
    end_date = f"{year}-{month+1}"
    date_index = pd.date_range(end=end_date, periods=12, freq='M')
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

def create_defect_summary_pohang(year:int,
                                 month:int,
                                 data:pd.DataFrame,
                                 months_window:tuple=(5,6,7),
                                 plant_name:str="포항") -> pd.DataFrame:
    """
    data 컬럼 가정:
      ['연도','월','구분1(공장)','구분2(제품군: CHQ/CD/...)','구분3(원인: 공정성/소재성)','실적']
    - 구분1==plant_name 만 사용
    - 컬럼: '24년 월평균', '25년 목표'(있으면), 선택월(예: 5,6,7), '합계', '월평균'
    """
    df = data.copy()
    # 기본 정규화
    for c in ['연도','월','실적']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['구분1','구분2','구분3']:
        if c not in df.columns: df[c] = ''
        df[c] = df[c].fillna('').astype(str)

    # 포항만
    df = df[df['구분1'].str.contains(plant_name)]
    if df.empty:
        return pd.DataFrame()

    # 편의
    prev_year = year - 1
    mlist = list(months_window)

    # 헬퍼: 안전 합/평균
    def safe_sum(s):  return float(np.nansum(s)) if len(s) else 0.0
    def safe_mean(s): return float(np.nanmean(s)) if len(s) else 0.0

    # 집계 함수
    def pick(g2=None, g3=None, yy=None, mm=None):
        q = df.copy()
        if g2 is not None: q = q[q['구분2']==g2]
        if g3 is not None: q = q[q['구분3']==g3]
        if yy is not None: q = q[q['연도']==yy]
        if mm is not None: q = q[q['월']==mm]
        return safe_sum(q['실적'])

    # 행 구성: (라벨A, 라벨B)
    rows = [
        ('정품','공정성'),
        ('정품','소재성'),
        ('CHQ',''),
        ('정품','공정성'),
        ('정품','소재성'),
        ('CD',''),
        ('공정성',''),   # 전체 공정성
        ('소재성',''),   # 전체 소재성
        ('포항','')      # 포항 총계
    ]

    # 컬럼 구성
    col_prev_avg = f"{str(prev_year)[-2:]}년 월평균"
    col_target   = f"{str(year)[-2:]}년 목표"
    month_cols   = [f"{m}월" for m in mlist]
    cols = [col_prev_avg, col_target] + month_cols + ['합계','월평균']

    out = pd.DataFrame(0.0, index=pd.MultiIndex.from_tuples(rows, names=['구분','']),
                       columns=cols)

    # --- ①  전년(전체 12개월) 월평균
    # CHQ/CD 내부 공정성/소재성
    for g2, base_row in [('CHQ',0), ('CD',3)]:
        prev_vals_ps = [pick(g2=g2, g3='공정성', yy=prev_year, mm=m) for m in range(1,13)]
        prev_vals_ms = [pick(g2=g2, g3='소재성', yy=prev_year, mm=m) for m in range(1,13)]
        out.iloc[base_row,   out.columns.get_loc(col_prev_avg)] = safe_mean(prev_vals_ps)
        out.iloc[base_row+1, out.columns.get_loc(col_prev_avg)] = safe_mean(prev_vals_ms)
        out.iloc[base_row+2, out.columns.get_loc(col_prev_avg)] = safe_mean([a+b for a,b in zip(prev_vals_ps, prev_vals_ms)])

    # 전체 공정성/소재성/포항
    prev_ps_all = [pick(g3='공정성', yy=prev_year, mm=m) for m in range(1,13)]
    prev_ms_all = [pick(g3='소재성', yy=prev_year, mm=m) for m in range(1,13)]
    out.loc[('공정성',''), col_prev_avg] = safe_mean(prev_ps_all)
    out.loc[('소재성',''), col_prev_avg] = safe_mean(prev_ms_all)
    out.loc[('포항',''),   col_prev_avg] = safe_mean([prev_ps_all[i]+prev_ms_all[i] for i in range(12)])

    # --- ②  목표(연/월 목표가 데이터에 있으면 사용, 없으면 0)
    #   목표가 별도 구분(예: 구분3=='목표')로 들어있는 경우를 지원
    def pick_target(g2=None, g3=None):
        q = df.copy()
        if '구분4' in df.columns:
            q = q[q['구분4']=='목표']
        else:
            q = q[q['구분3']=='목표']  # 없으면 0으로 떨어짐
        if g2 is not None: q = q[q['구분2']==g2]
        if g3 is not None: q = q[q['구분3']==g3]
        if q.empty: return 0.0
        return safe_sum(q['실적'])

    # 필요시 0으로 채움(목표 데이터가 없다는 전제)
    # 여기선 일단 0으로 둔다.
    # out.loc[:, col_target] = 0.0

    # --- ③  당해 선택월, 합계/월평균
    def fill_block(g2, base_row):
        # 공정성/소재성 개별
        for idx, cause in enumerate(['공정성','소재성']):
            vals = [pick(g2=g2, g3=cause, yy=year, mm=m) for m in mlist]
            out.iloc[base_row+idx, out.columns.get_indexer(month_cols)] = vals
            out.iloc[base_row+idx, out.columns.get_loc('합계')]   = safe_sum(vals)
            out.iloc[base_row+idx, out.columns.get_loc('월평균')] = safe_mean(vals)
        # 합계행(CHQ/CD)
        vals_sum = [pick(g2=g2, yy=year, mm=m) for m in mlist]
        out.iloc[base_row+2, out.columns.get_indexer(month_cols)] = vals_sum
        out.iloc[base_row+2, out.columns.get_loc('합계')]   = safe_sum(vals_sum)
        out.iloc[base_row+2, out.columns.get_loc('월평균')] = safe_mean(vals_sum)

    fill_block('CHQ', 0)
    fill_block('CD',  3)

    # 전체 공정성/소재성 + 포항 총계
    vals_ps_all = [pick(g3='공정성', yy=year, mm=m) for m in mlist]
    vals_ms_all = [pick(g3='소재성', yy=year, mm=m) for m in mlist]
    out.loc[('공정성',''), month_cols] = vals_ps_all
    out.loc[('소재성',''), month_cols] = vals_ms_all
    out.loc[('공정성',''), ['합계','월평균']] = [safe_sum(vals_ps_all), safe_mean(vals_ps_all)]
    out.loc[('소재성',''), ['합계','월평균']] = [safe_sum(vals_ms_all), safe_mean(vals_ms_all)]

    vals_total = [vals_ps_all[i] + vals_ms_all[i] for i in range(len(mlist))]
    out.loc[('포항',''), month_cols] = vals_total
    out.loc[('포항',''), ['합계','월평균']] = [safe_sum(vals_total), safe_mean(vals_total)]

    # 반올림/형식
    out = out.round(0)

    return out




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




# ====================== 재무상태표(연결) ======================


def _bs_paren_to_signed(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.replace(",", "", regex=False).str.strip()
    neg = s.str.match(r"^\(.*\)$")
    s = s.str.replace(r"[\(\)]", "", regex=True)
    v = pd.to_numeric(s, errors="coerce")
    v[neg] = -v[neg].abs()
    return v.fillna(0.0)

def _bs_norm_company(x: str) -> str:
    x = str(x).strip()
    return "태국" if x in ("타이", "태국") else x

def clean_bs_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    item_col = next((c for c in ["구분","구분3","항목"] if c in df.columns), None)
    if item_col is None:
        raise ValueError("재무상태 데이터에 '구분'(또는 '구분3'/'항목')이 필요합니다.")
    comp_col = next((c for c in ["구분2","회사","법인"] if c in df.columns), None)
    if comp_col is None:
        comp_col = "_회사"; df[comp_col] = "전체"

    for c in ["연도","월","실적"]:
        if c not in df.columns:
            raise ValueError(f"재무상태표 데이터에 '{c}' 컬럼이 필요합니다.")

    df[item_col] = df[item_col].astype(str).str.strip().str.replace(r"\s+"," ", regex=True)
    df[comp_col] = df[comp_col].astype(str).str.strip().str.replace(r"\s+"," ", regex=True).map(_bs_norm_company)
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    df["월"]   = pd.to_numeric(df["월"],   errors="coerce").astype("Int64")
    df["실적"] = _bs_paren_to_signed(df["실적"])

    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols: df = df.drop(columns=drop_cols, errors="ignore")
    return df.rename(columns={item_col: "구분", comp_col: "회사"})

def create_bs_snapshot_by_gubun(year: int, month: int, data: pd.DataFrame) -> pd.DataFrame:
    """
    반환: index='구분', columns = [''24',''25','당월', <회사들...>, '전월비 증감']
      '24'        : 전년도 마지막 가용월 값
      '25'        : 금년도 전월 값(1월이면 전년도 마지막 월)
      당월        : 금년도 해당월 합계
      회사들      : 금년도 해당월 각 법인 값
      전월비 증감 : 당월 - '25'
    """
    df = clean_bs_df(data)

    # 구분 순서 보존
    gubun_order = list(dict.fromkeys(df["구분"].astype(str).tolist()))

    # 전년도 마지막 가용월
    prev_year_avail = sorted(df.loc[df["연도"]==year-1, "월"].dropna().unique())
    prev_year_last_month = int(prev_year_avail[-1]) if prev_year_avail else 12

    # 금년도 가용월 & 폴백(선택월이 없으면 가까운 과거월)
    this_avail = sorted(df.loc[df["연도"]==year, "월"].dropna().unique())
    used_month = month
    if this_avail and month not in this_avail:
        past = [m for m in this_avail if m <= month]
        used_month = int(max(past) if past else max(this_avail))

    prev_month = used_month - 1
    prev_month_year = year
    if prev_month < 1:
        prev_month = prev_year_last_month
        prev_month_year = year - 1

    # 집계함수(중복 안전)
    def items_at(y, m):
        s = df[(df["연도"]==y) & (df["월"]==m)].groupby("구분", sort=False)["실적"].sum()
        if s.index.duplicated().any(): s = s.groupby(level=0).sum()
        return s

    def items_company_at(y, m):
        pv = (df[(df["연도"]==y) & (df["월"]==m)]
              .pivot_table(index="구분", columns="회사", values="실적",
                           aggfunc="sum", fill_value=0.0, observed=False))
        if pv.index.duplicated().any(): pv = pv.groupby(level=0).sum()
        # 표에 보여줄 회사 순서 선호
        prefer = ["특수강","본사","남통","천진","태국"]
        exists = [c for c in prefer if c in pv.columns]
        others = [c for c in pv.columns if c not in exists]
        return pv[exists + others]

    col_prev_year = items_at(year-1, prev_year_last_month)     # '24
    col_prev      = items_at(prev_month_year, prev_month)      # '25
    col_curr      = items_at(year, used_month)                 # 당월
    by_comp       = items_company_at(year, used_month)         # 당월(법인)

    all_items = pd.Index(gubun_order, name="구분")
    out = pd.DataFrame(index=all_items, dtype=float)
    out["'24"] = col_prev_year.reindex(all_items).fillna(0.0).values
    out["'25"] = col_prev.reindex(all_items).fillna(0.0).values
    out["당월"] = col_curr.reindex(all_items).fillna(0.0).values

    for c in by_comp.columns:
        out[c] = by_comp.reindex(all_items)[c].fillna(0.0).values

    out["전월비 증감"] = out["당월"] - out["'25"]
    comp_cols = [c for c in out.columns if c not in ["'24","'25","당월","전월비 증감"]]
    out = out[["'24","'25","당월"] + comp_cols + ["전월비 증감"]]
    return out

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
    main_items: tuple[str, ...] = ("CHQ","CD","STS","BTB","PB","내수","수출"),
    filter_tag: str = "수정손익"
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