import subprocess
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "outputs" / "predictions.csv"
INFERENCE_SCRIPT = BASE_DIR / "inference.py"


st.header("X-Sell Model Predictions")

if st.button("Обновить", type="primary"):
    with st.spinner("Обновление предсказаний…"):
        result = subprocess.run(
            [sys.executable, str(INFERENCE_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
    if result.returncode == 0:
        st.cache_data.clear()
        st.success("Предсказания обновлены")
        st.rerun()
    else:
        st.error("Ошибка обновления предсказаний")
        st.code(result.stderr or result.stdout)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=",")


df_raw = load_data()


st.sidebar.header("Фильтры")

cat_filters = {
    "Канал привлечения": "who_is_this_first_inn",
    "Статус оттока": "inn_status",
    "Сегмент": "current_segment_inn",
    "Группа MCC": "top_mcc_group_inn",
}

cat_selections: dict[str, list] = {}
for label, col in cat_filters.items():
    cat_selections[col] = st.sidebar.multiselect(
        label, sorted(df_raw[col].dropna().unique())
    )

selected_products = st.sidebar.multiselect(
    "Продукт (relevant_product)",
    sorted(df_raw["relevant_product"].dropna().unique()),
)


def make_slider(label: str, col: str) -> tuple[float, float]:
    lo = float(df_raw[col].min())
    hi = float(df_raw[col].max())
    if lo == hi:
        return lo, hi
    return st.sidebar.slider(label, min_value=lo, max_value=hi, value=(lo, hi), format="%.2f")


ev_range = make_slider("Expected Value", "expected_value")
turnover_range = make_slider("Turnover MA 3M", "turnover_ma_3m")
cnt_range = make_slider("Cnt Trx MA 3M", "cnt_trx_ma_3m")
avg_check_range = make_slider("Avg Check WMA 3M", "avg_check_wma_3m")

n_rows = st.sidebar.number_input(
    "Кол-во строк для отображения",
    min_value=1,
    max_value=len(df_raw),
    value=2000,
    step=100,
)


mask = pd.Series(True, index=df_raw.index)

for col, selected in cat_selections.items():
    if selected:
        mask &= df_raw[col].isin(selected)

if selected_products:
    mask &= df_raw["relevant_product"].isin(selected_products)

mask &= df_raw["expected_value"].between(*ev_range)
mask &= df_raw["turnover_ma_3m"].between(*turnover_range)
mask &= df_raw["cnt_trx_ma_3m"].between(*cnt_range)
mask &= df_raw["avg_check_wma_3m"].between(*avg_check_range)

df_filtered = df_raw.loc[mask].sort_values("expected_value", ascending=False)


df_display = df_filtered.head(int(n_rows))

st.caption(f"Показано {len(df_display)} из {len(df_filtered)} строк")
st.dataframe(df_display, use_container_width=True)


st.divider()

col_metric, col_chart = st.columns([1, 2])

with col_metric:
    st.metric(
        label="Expected Value",
        value=f"{df_display['expected_value'].sum():,.2f}",
    )

with col_chart:
    ev_by_product = (
        df_display.groupby("relevant_product", as_index=False)["expected_value"]
        .sum()
        .sort_values("expected_value", ascending=False)
    )
    fig = px.bar(
        ev_by_product,
        x="relevant_product",
        y="expected_value",
        labels={"relevant_product": "Продукт", "expected_value": "Expected Value"},
        title="Expected Value по продуктам",
        text_auto=".2s",
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)


st.download_button(
    label="Скачать",
    data=df_display.to_csv(index=False, sep=";"),
    file_name="predictions_filtered.csv",
    mime="text/csv",
)
