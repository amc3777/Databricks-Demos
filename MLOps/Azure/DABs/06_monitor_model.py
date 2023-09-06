# Databricks notebook source
# MAGIC %pip install -q "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.3.0-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import databricks.lakehouse_monitoring as lm
import mlflow

CATALOG = "andrewcooleycatalog"
SCHEMA = "andrewcooleyschema"

mlflow.set_registry_uri('databricks-uc')

try:
  
  info = lm.get_monitor(table_name=f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_inferencelogs")

except Exception:

  enabled = False
  print("Lakehouse Monitoring is still in gated public preview. Please enroll this account in the preview to use the product.")
