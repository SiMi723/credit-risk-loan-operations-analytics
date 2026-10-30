-- schema_validation.sql
-- Quick schema and key-overlap checks. Run against DuckDB interactively or
-- via scripts/01_ingest.py.

-- 1. Row counts per table
SELECT 'application_train' AS tbl, COUNT(*) AS n_rows
FROM read_csv_auto('data/raw/application_train.csv', ignore_errors=true)
UNION ALL
SELECT 'bureau', COUNT(*)
FROM read_csv_auto('data/raw/bureau.csv', ignore_errors=true)
UNION ALL
SELECT 'previous_application', COUNT(*)
FROM read_csv_auto('data/raw/previous_application.csv', ignore_errors=true)
UNION ALL
SELECT 'installments_payments', COUNT(*)
FROM read_csv_auto('data/raw/installments_payments.csv', ignore_errors=true);

-- 2. Target distribution in application_train
SELECT TARGET, COUNT(*) AS n, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM read_csv_auto('data/raw/application_train.csv', ignore_errors=true)
GROUP BY TARGET
ORDER BY TARGET;

-- 3. Key overlap: bureau ∩ application_train
SELECT
    COUNT(DISTINCT b.SK_ID_CURR)   AS bureau_unique_ids,
    COUNT(DISTINCT a.SK_ID_CURR)   AS app_unique_ids,
    COUNT(DISTINCT CASE WHEN a.SK_ID_CURR IS NOT NULL THEN b.SK_ID_CURR END) AS overlap
FROM read_csv_auto('data/raw/bureau.csv', ignore_errors=true) b
LEFT JOIN (
    SELECT DISTINCT SK_ID_CURR
    FROM read_csv_auto('data/raw/application_train.csv', ignore_errors=true)
) a ON b.SK_ID_CURR = a.SK_ID_CURR;
