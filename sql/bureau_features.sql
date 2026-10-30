-- bureau_features.sql
-- Aggregate bureau.csv to SK_ID_CURR grain.
-- Run via DuckDB: duckdb -c "$(cat sql/bureau_features.sql)"
-- Or executed programmatically via src/credit_risk/features.py

SELECT
    SK_ID_CURR,
    COUNT(*)                                                    AS bureau_credit_count,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)   AS bureau_active_count,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END)   AS bureau_closed_count,
    MAX(CREDIT_DAY_OVERDUE)                                     AS bureau_overdue_max,
    AVG(CREDIT_DAY_OVERDUE)                                     AS bureau_overdue_mean,
    SUM(AMT_CREDIT_SUM)                                         AS bureau_amt_credit_sum,
    AVG(AMT_CREDIT_SUM)                                         AS bureau_amt_credit_mean,
    MIN(DAYS_CREDIT)                                            AS bureau_days_credit_min,
    MAX(DAYS_CREDIT)                                            AS bureau_days_credit_max,
    AVG(DAYS_CREDIT)                                            AS bureau_days_credit_mean,
    SUM(AMT_CREDIT_SUM_DEBT)                                    AS bureau_debt_sum,
    AVG(AMT_CREDIT_SUM_DEBT)                                    AS bureau_debt_mean,
    SUM(AMT_ANNUITY)                                            AS bureau_annuity_sum,
    COUNT(DISTINCT CREDIT_TYPE)                                 AS bureau_credit_type_count
FROM read_csv_auto('data/raw/bureau.csv', ignore_errors=true)
GROUP BY SK_ID_CURR
ORDER BY SK_ID_CURR;
