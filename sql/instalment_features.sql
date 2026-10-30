-- instalment_features.sql
-- Aggregate installments_payments.csv (13.6M rows) to SK_ID_CURR grain.
-- All aggregation is performed in DuckDB to avoid loading full table into memory.
--
-- Feature definitions:
--   dpd (days past due)       = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
--                               positive → payment was late
--                               negative → payment was early
--   amt_diff                  = AMT_PAYMENT - AMT_INSTALMENT
--                               negative → underpaid, positive → overpaid
--
-- Missing/invalid handling:
--   Rows where AMT_INSTALMENT IS NULL or AMT_PAYMENT IS NULL are excluded
--   from aggregation (they cannot contribute meaningful payment behaviour).

WITH base AS (
    SELECT
        SK_ID_CURR,
        DAYS_INSTALMENT,
        DAYS_ENTRY_PAYMENT,
        AMT_INSTALMENT,
        AMT_PAYMENT,
        (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)          AS dpd,
        (AMT_PAYMENT - AMT_INSTALMENT)                  AS amt_diff
    FROM read_csv_auto('data/raw/installments_payments.csv', ignore_errors=true)
    WHERE AMT_INSTALMENT IS NOT NULL
      AND AMT_PAYMENT    IS NOT NULL
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                                AS instalment_count,
    SUM(CASE WHEN dpd > 0 THEN 1 ELSE 0 END)               AS instalment_late_count,
    AVG(GREATEST(dpd, 0))                                  AS instalment_dpd_mean,
    MAX(GREATEST(dpd, 0))                                  AS instalment_dpd_max,
    SUM(CASE WHEN dpd > 0 THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(COUNT(*), 0)                              AS instalment_late_ratio,
    AVG(amt_diff)                                          AS instalment_amt_diff_mean,
    MIN(amt_diff)                                          AS instalment_amt_diff_min,
    SUM(CASE WHEN amt_diff < -1 THEN 1 ELSE 0 END)         AS instalment_underpay_count,
    AVG(AMT_PAYMENT)                                       AS instalment_amt_payment_mean,
    AVG(AMT_INSTALMENT)                                    AS instalment_amt_scheduled_mean
FROM base
GROUP BY SK_ID_CURR
ORDER BY SK_ID_CURR;
