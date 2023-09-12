# Databricks notebook source
import mlflow
from mlflow.models.signature import infer_signature
import os
import h2o
from pysparkling import *
from pysparkling.ml import H2OExtendedIsolationForest
from pyspark.sql.functions import col
from pyspark.sql import functions as F
import mlflow.pyfunc
import platform
from mlflow.models.signature import infer_signature

# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.linalg import SparseVector
# import pandas as pd
# import numpy as np

# COMMAND ----------

hc = H2OContext.getOrCreate()

# COMMAND ----------

h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")
sparkDF = hc.asSparkFrame(h2o_df)
[trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])

# COMMAND ----------

predictors = ["AGE", "RACE", "DPROS", "DCAPS", "PSA", "VOL", "GLEASON"]

algo = H2OExtendedIsolationForest(
                                  featuresCols=predictors,
                                  sampleSize=256,
                                  ntrees=100,
                                  seed=123,
                                  extensionLevel=len(predictors) - 1)

model = algo.fit(trainingDF)

# COMMAND ----------

model.getModelSummary()

# COMMAND ----------

model.transform(testingDF).show(truncate = False)

# COMMAND ----------

try:

  h2o_model_path = "/Volumes/andrewcooleycatalog/andrewcooleyschema/managedvolume"
  model.save(h2o_model_path)

except Exception:
  
  print("Model already exists.")

# COMMAND ----------

artifacts = {"h2o_model": f"{h2o_model_path}/mojo_model"}

# COMMAND ----------

uploaded_model = h2o.upload_mojo(artifacts["h2o_model"])

# COMMAND ----------

uploaded_model.predict(h2o.H2OFrame(testingDF))

# COMMAND ----------

class H2OWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
      
        import h2o

        h2o.init()

        self.h2o_model = h2o.upload_mojo(context.artifacts["h2o_model"])

    def predict(self, context, model_input, params=None):

        input_h2oframe = h2o.H2OFrame(model_input)
        output_h2oframe = self.h2o_model.predict(input_h2oframe)
        predictions = output_h2oframe.as_data_frame()

        return predictions

# COMMAND ----------

PYTHON_VERSION = platform.python_version()
import requests
import tabulate
import future
import h2o

conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python={}".format(PYTHON_VERSION),
        "h2o-py", 
        "openjdk",
        "pip",
        {
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "requests=={}".format(requests.__version__),
                "tabulate=={}".format(tabulate.__version__),
                "future=={}".format(future.__version__)
            ],
        },
    ],
    "name": "h2o_env",
}

# COMMAND ----------

uploaded_model = h2o.upload_mojo(artifacts["h2o_model"])

input_example = testingDF.limit(5).toPandas()
output_example = uploaded_model.predict(h2o.H2OFrame(testingDF.limit(5).toPandas())).as_data_frame()
display(input_example)
display(output_example)

# COMMAND ----------

with mlflow.start_run() as run:

  mlflow.pyfunc.log_model(
  artifact_path="ExtendedIsolationForestModel",
  conda_env=conda_env,
  python_model=H2OWrapper(),
  artifacts=artifacts,
  registered_model_name="ExtendedIsolationForestModel",
  signature=infer_signature(input_example, output_example),
  input_example=input_example
  )
  
mlflow.end_run()

# COMMAND ----------

import mlflow

logged_model = f'runs:/{run.info.run_id}/ExtendedIsolationForestModel'

loaded_model = mlflow.pyfunc.load_model(logged_model)

test_predictions = loaded_model.predict(input_example)

print(test_predictions)
