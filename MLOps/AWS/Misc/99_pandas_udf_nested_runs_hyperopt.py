# Databricks notebook source
import pyspark.sql.functions as f
from pyspark.sql.functions import struct, col
import mlflow
from hyperopt import fmin, hp, tpe, Trials
import time as time
import numpy as np
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate total records for 10 forecasting groups in a Spark dataframe

# COMMAND ----------

df = (spark
      .range(100000000)
      .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("group_id"))
      .withColumn("feature_1", f.rand() * 1)
      .withColumn("feature_2", f.rand() * 2)
      .withColumn("feature_3", f.rand() * 3)
      .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())
     )

display(df)
print(f'{df.count()} records in the dataset')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create an empty MLflow Experiment to track training runs with

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = "/Users/{}/pandas_udf_hyperopt".format(user)

try:

  experiment_id = mlflow.create_experiment(
      experiment_name
  )

except Exception:

  search_query_string = "name ='{}'".format(experiment_name)
  experiment_id = mlflow.search_experiments(filter_string=search_query_string)[0].experiment_id
  print("Experiment already exists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create model training function for each group's nested run and use HyperOpt for parameter tuning for each group's model

# COMMAND ----------

mlflow.autolog(log_models=False)

search_space = {
    "l1_ratio": hp.quniform("l1_ratio", 0, 1, 0.1)
}

num_evals = 3
trials = Trials()

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:

    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]
    
    group_id = df_pandas["group_id"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0]

    def objective_function(params):    
      
      l1_ratio = params["l1_ratio"]

      with mlflow.start_run(run_name="trial-"+str(time.time()), nested=True, experiment_id=experiment_id):
        
        enet = ElasticNet(l1_ratio=l1_ratio)
        enet.fit(X, y)

        predictions = enet.predict(X)
        mse = mean_squared_error(y, predictions)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("mse", mse)

      return mse
   
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as outer_run:

        with mlflow.start_run(run_name="group "+str(group_id), nested=True, experiment_id=experiment_id) as run:

          best_hp = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))

          enet = ElasticNet(l1_ratio=best_hp["l1_ratio"])
          enet.fit(X, y)

          predictions = enet.predict(X)
          mse = mean_squared_error(y, predictions)

          mlflow.sklearn.log_model(enet, str(group_id))
          mlflow.log_param("l1_ratio", best_hp["l1_ratio"])
          mlflow.log_metric("mse", mse)
          mlflow.set_tag("group", str(group_id))

          artifact_uri = f"runs:/{run.info.run_id}/{group_id}"

          return_df = pd.DataFrame([[group_id, n_used, artifact_uri, mse]], 
                                    columns=["group_id", "n_used", "model_path", "mse"])

    return return_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create parent run and output dataframe from applied function

# COMMAND ----------

train_return_schema = "group_id integer, n_used integer, model_path string, mse float"

with mlflow.start_run(run_name="Training run for all groups", experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    model_directories_df = (df
        .withColumn("run_id", f.lit(run_id))
        .groupby("group_id")
        .applyInPandas(train_model, schema=train_return_schema)
    )

combined_df = df.join(model_directories_df, on="group_id", how="left")
display(combined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a custom Python function model to load and predict with a distinct model for each group

# COMMAND ----------

from mlflow.pyfunc import PythonModel

experiment_id = run.info.experiment_id

model_df = (spark.read.format("mlflow-experiment")
            .load(experiment_id)
            .filter("tags.group IS NOT NULL")
            .orderBy("end_time", ascending=False)
            .select("tags.group", "run_id")
            .limit(10))

group_to_model = {row["group"]: mlflow.sklearn.load_model(f"runs:/{row['run_id']}/{row['group']}") for row in model_df.collect()}
                                                        
class GroupModel(PythonModel):
    
    def __init__(self, group_to_model_map):
        self.group_to_model_map = group_to_model_map
        
    def predict_for_group(self, row):
        '''
        This method applies to a single row of data by
        fetching the appropriate model and generating predictions
        '''
        model = self.group_to_model_map.get(str(row["group_id"]))
        data = row[["feature_1", "feature_2", "feature_3"]].to_frame().T
        return model.predict(data)[0]
    
    def predict(self, model_input):
        return model_input.apply(self.predict_for_group, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call predict method of custom model object

# COMMAND ----------

example_model = GroupModel(group_to_model)
display(example_model.predict(combined_df.toPandas().head(5)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log model to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="custom model", experiment_id=experiment_id) as run:
    model = GroupModel(group_to_model)
    mlflow.pyfunc.log_model("model", python_model=model)
    model_uri = f"runs:/{run.info.run_id}/model"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Many grouped forecasting models on Databricks:
# MAGIC
# MAGIC 1. <a href="https://docs.databricks.com/en/machine-learning/automl/index.html" target="_blank">AutoML for Forecasting</a>
# MAGIC
# MAGIC 2. <a href="https://github.com/databricks/diviner" target="_blank">Diviner</a>
# MAGIC
# MAGIC 3. <a href="https://www.databricks.com/solutions/accelerators/demand-forecasting" target="_blank">Solution Accelerators for Fine-grained Forecasting</a>
# MAGIC

# COMMAND ----------


