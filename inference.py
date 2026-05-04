import joblib
import pandas as pd
from acquiring_xsell.data import load_data
from acquiring_xsell.preprocessing import process_dtypes, fill_missing_values
from acquiring_xsell.features import build_features, make_product_df

OUTPUT_COLS = [
    'inn',
    'inn_status',
    'first_trx_month_inn',
    'current_segment_inn',
    'top_mcc_group_inn',
    'who_is_this_first_inn',
    'real_kam_on_inn',
    'turnover_ma_3m',
    'cnt_trx_ma_3m',
    'avg_check_wma_3m',
    'relevant_product',
    'expected_value',
]

PRE_SCALED_FEATURES = [
    'cnt_trx',
    'turnover',
    'avg_check',
    'turnover_ma_2m',
    'turnover_ema_2m',
    'turnover_ma_3m',
    'turnover_ema_3m',
    'turnover_ma_6m',
    'turnover_ema_6m',
    'cnt_trx_ma_2m',
    'cnt_trx_ema_2m',
    'cnt_trx_ma_3m',
    'cnt_trx_ema_3m',
    'cnt_trx_ma_6m',
    'cnt_trx_ema_6m',
    'avg_check_wma_2m',
    'avg_check_wma_3m',
    'avg_check_wma_6m',
    'altpay1_turnover',
    'altpay2_turnover',
    'altpay3_turnover',
    'altpay4_turnover',
    'altpay5_turnover',
    'altpay6_turnover',
]

P1P2_VALUE = 40_000
P3_VALUE = 15_000
ALTPAY5_VALUE = 25_000


def main():
    df = load_data()
    df = process_dtypes(df)
    df = fill_missing_values(df)
    df = build_features(df, drop_dttm_cols=False)

    max_month = df['month'].max()

    df_p1p2 = make_product_df(df, "p1p2")
    df_p3 = make_product_df(df, "p3")
    df_altpay5 = make_product_df(df, "altpay5")

    p3_proba_in_rule = df_p3[df_p3['rule_flag_p3'] == 1]['target'].mean()
    p3_proba_out_rule = df_p3[df_p3['rule_flag_p3'] == 0]['target'].mean()

    df_p1p2 = df_p1p2[df_p1p2['month'] == max_month]
    df_p3 = df_p3[df_p3['month'] == max_month]
    df_altpay5 = df_altpay5[df_altpay5['month'] == max_month]

    model_p1p2 = joblib.load("models/logreg_p1p2_wo_scaled_features.joblib")
    model_altpay5 = joblib.load("models/logreg_altpay5_wo_scaled_features.joblib")

    df_p1p2['proba'] = model_p1p2.predict_proba(df_p1p2)[:, 1]
    df_p3.loc[df_p3['rule_flag_p3'] == 1, 'proba'] = p3_proba_in_rule
    df_p3.loc[df_p3['rule_flag_p3'] == 0, 'proba'] = p3_proba_out_rule
    df_altpay5['proba'] = model_altpay5.predict_proba(df_altpay5)[:, 1]

    df_p1p2['expected_value'] = df_p1p2['proba'] * P1P2_VALUE
    df_p3['expected_value'] = df_p3['proba'] * P3_VALUE
    df_altpay5['expected_value'] = df_altpay5['proba'] * ALTPAY5_VALUE


    df_p1p2['relevant_product'] = 'P1P2'
    df_p3['relevant_product'] = 'P3'
    df_altpay5['relevant_product'] = 'Altpay5'

    df_all_products = pd.concat([
        df_p1p2,
        df_p3,
        df_altpay5
    ])


    df_out = df_all_products[OUTPUT_COLS].copy()

    df_out.to_csv("outputs/predictions.csv", index=False)


if __name__ == "__main__":
    main()