-- Databricks notebook source
-- MAGIC %md
-- MAGIC A notebook that provides an example Delta Live Tables pipeline to:
-- MAGIC
-- MAGIC - Read raw JSON clickstream data into a streaming table.
-- MAGIC - Read records from the raw data table and use a Delta Live Tables query to create a new streaming table with cleaned and prepared data.
-- MAGIC - Perform an analysis on the prepared data with a Delta Live Tables query and materialized view.

-- COMMAND ----------

-- DBTITLE 1,Ingest raw clickstream data
CREATE OR REFRESH STREAMING LIVE TABLE clickstream_raw
COMMENT "The raw wikipedia click stream dataset, ingested from /databricks-datasets."
AS SELECT * FROM cloud_files("/databricks-datasets/wikipedia-datasets/data-001/clickstream/raw-uncompressed-json/", "json")

-- COMMAND ----------

-- DBTITLE 1,Clean and prepare data
CREATE OR REFRESH STREAMING LIVE TABLE clickstream_clean
COMMENT "Wikipedia clickstream data cleaned and prepared for analysis."
AS SELECT
  curr_title AS current_page_title,
  CAST(n AS INT) AS click_count,
  prev_title AS previous_page_title
FROM STREAM(LIVE.clickstream_raw);

-- COMMAND ----------

-- DBTITLE 1,Top referring pages
CREATE OR REFRESH LIVE TABLE top_spark_referers
COMMENT "A table containing the top pages linking to the Apache Spark page."
AS SELECT
  previous_page_title as referrer,
  click_count
FROM LIVE.clickstream_clean
WHERE current_page_title = 'Apache_Spark'
ORDER BY click_count DESC
LIMIT 10
