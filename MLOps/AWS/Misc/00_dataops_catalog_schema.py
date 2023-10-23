# Databricks notebook source
# MAGIC %md
# MAGIC ### Notebook set-up steps

# COMMAND ----------

import warnings

warnings.filterwarnings("ignore")

dbutils.widgets.removeAll()

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
uc_prefix = user.replace(".", "").split("@")[0]
catalog = uc_prefix + "_catalog"
schema = uc_prefix + "_schema"
volume = uc_prefix + "_managedvolume"

dbutils.widgets.text("catalog", catalog)
dbutils.widgets.text("schema", schema)
dbutils.widgets.text("volume",volume)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create objects in Unity Catalog for downstream workflows

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS ${catalog};
# MAGIC USE CATALOG ${catalog};
# MAGIC CREATE SCHEMA IF NOT EXISTS ${schema};
# MAGIC USE SCHEMA ${schema};
# MAGIC CREATE VOLUME IF NOT EXISTS ${volume};
