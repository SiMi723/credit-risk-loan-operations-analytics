-- prev_app_features.sql
-- Aggregate previous_application.csv to SK_ID_CURR grain.

SELECT
    SK_ID_CURR,
    COUNT(*)                                                              AS prev_app_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)   AS prev_approved_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END)   AS prev_refused_count,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1 ELSE 0 END)   AS prev_canceled_count,
    AVG(AMT_CREDIT)                                                       AS prev_amt_credit_mean,
    MAX(AMT_CREDIT)                                                       AS prev_amt_credit_max,
    AVG(AMT_ANNUITY)                                                      AS prev_annuity_mean,
    AVG(AMT_APPLICATION)                                                  AS prev_amt_application_mean,
    AVG(DAYS_DECISION)                                                    AS prev_days_decision_mean,
    MIN(DAYS_DECISION)                                                    AS prev_days_decision_min,
    AVG(CASE WHEN AMT_APPLICATION > 0
             THEN AMT_CREDIT / AMT_APPLICATION ELSE NULL END)             AS prev_credit_to_app_ratio,
    COUNT(DISTINCT NAME_CONTRACT_TYPE)                                    AS prev_contract_type_count,
    SUM(CASE WHEN NAME_YIELD_GROUP = 'high'  THEN 1 ELSE 0 END)           AS prev_high_yield_count
FROM read_csv_auto('data/raw/previous_application.csv', ignore_errors=true)
GROUP BY SK_ID_CURR
ORDER BY SK_ID_CURR;
