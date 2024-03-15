# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

import mlflow
print(mlflow.__version__)

# COMMAND ----------

# MAGIC %pip install -U databricks-sdk
# MAGIC %pip install -U mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
print(mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC Set variables for use as parameters to the model's deployment - these can be updated in the widgets at the top of the notebook UI.

# COMMAND ----------

model_names = ['mixtral_8x7B_Instruct_v0_1', 'mixtral_8x7b_v0_1']
dbutils.widgets.dropdown("model_name", model_names[0], model_names)

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
uc_prefix = user.replace(".", "").split("@")[0]
catalog = uc_prefix + "_marketplacemodels_2"
dbutils.widgets.text("catalog_name", catalog)


version = "1"
dbutils.widgets.text("model_version", version)

version_spark = "1"
dbutils.widgets.text("model_version_spark", version_spark)
model_version_spark = dbutils.widgets.get("model_version_spark")

model_name = dbutils.widgets.get("model_name")
catalog_name = dbutils.widgets.get("catalog_name")


endpoint = f"{catalog_name}_{model_name}"
dbutils.widgets.text("endpoint_name", endpoint)

model_version = dbutils.widgets.get("model_version")
endpoint_name = dbutils.widgets.get("endpoint_name")
registered_model_name = f"{catalog_name}.models.{model_name}"

dbutils.widgets.text("min_provisioned_throughput", "1")
dbutils.widgets.text("max_provisioned_throughput", "100")
min_provisioned_throughput = dbutils.widgets.get("min_provisioned_throughput")
max_provisioned_throughput = dbutils.widgets.get("max_provisioned_throughput")

# COMMAND ----------

import requests
import json

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

optimizable_info = requests.get(
    url=f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{registered_model_name}/{model_version}",
    headers=headers).json()

if 'optimizable' not in optimizable_info or not optimizable_info['optimizable']:
   raise ValueError("Model is not eligible for provisioned throughput")

chunk_size = optimizable_info['throughput_chunk_size']

print(chunk_size)

# COMMAND ----------

# MAGIC %md
# MAGIC It is recommended to set throughput parameters based on printed chunk size above. The text widgets above are set to 1 - 100 tok/sec as a very small sized default.

# COMMAND ----------

data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": registered_model_name,
                "model_version": model_version,
                "min_provisioned_throughput": min_provisioned_throughput,
                "max_provisioned_throughput": max_provisioned_throughput,
            }
        ]
    },
}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

logged_model = f"models:/{catalog_name}.models.{model_name}/{model_version_spark}"
generate = mlflow.pyfunc.spark_udf(spark, logged_model, "string")

# COMMAND ----------

import pandas as pd

data = {
    'messages': [
        [{'content': 'What is ML?', 'role': 'user'}]
    ]
}

df = spark.createDataFrame(pd.DataFrame(data))

print(df)

# COMMAND ----------

# You can use the UDF directly on a text column
generated_df = df.select(generate(df.messages).alias('generated_text'))

# COMMAND ----------

display(generated_df)
