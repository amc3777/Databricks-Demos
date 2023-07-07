# Databricks notebook source
import mlflow
mlflow.autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,
    disable=True,
    exclusive=True,
    disable_for_unsupported_versions=True,
    silent=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read heart disease (Kaggle) dataset into Spark dataframe

# COMMAND ----------

import os
df = spark.read.csv(f"file:{os.getcwd()}/dataset_heart.csv", header=True)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add ID column and replace the binary classification labels

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

df = df.withColumn("patient_id", monotonically_increasing_id()).replace("1", "0").replace("2", "1")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use databricks-automl-runtime package to create AutoML experiment programmatically

# COMMAND ----------

from databricks import automl

summary = automl.regress(
  dataset=training_set,
  target_col="heart disease",
  # data_dir: Optional[str] = None,
  exclude_columns=patient_id,
  # exclude_frameworks: Optional[List[str]] = None,                   # <DBR> 10.3 ML and above
  # experiment_dir: Optional[str] = None,                             # <DBR> 10.4 LTS ML and above
  experiment_name="AutoML-class_" + str(uuid.uuid4())[:6],
  # feature_store_lookups: Optional[List[Dict]] = None,               # <DBR> 11.3 LTS ML and above
  # imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None, # <DBR> 10.4 LTS ML and above
  # max_trials: Optional[int] = None,
  pos_label=1,                                 # <DBR> 10.5 ML and below
  primary_metric="accuracy",
  # time_col: Optional[str] = None,
  timeout_minutes=30
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition model version to staging

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Load model as MLflow Python function flavor for framework-agnostic & light-weight serving

# COMMAND ----------



# COMMAND ----------


