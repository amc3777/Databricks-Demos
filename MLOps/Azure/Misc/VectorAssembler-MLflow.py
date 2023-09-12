# Databricks notebook source
import pyspark
import mlflow
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

df = spark.createDataFrame([(1, 0, 3)], ["a", "b", "c"])
vecAssembler = VectorAssembler(outputCol="features")
vecAssembler.setInputCols(["a", "b", "c"])
df = vecAssembler.transform(df)
display(df)

# COMMAND ----------

vectorAssemblerPath = "dbfs/Volumes/andrewcooleycatalog/andrewcooleyschema/managedvolume/vector-assembler"
vecAssembler.write().overwrite().save(vectorAssemblerPath)
loadedAssembler = VectorAssembler.load(vectorAssemblerPath)

# COMMAND ----------

artifacts = {"vector_assembler": vectorAssemblerPath}

# COMMAND ----------

class ModelWrapper(mlflow.pyfunc.PythonModel):

  def predict(self, context, model_input, params=None):

    from pyspark.ml.feature import VectorAssembler

    vecAssembler = VectorAssembler.load(artifacts["vector_assembler"])

    model_input = vecAssembler.transform(model_input)

    return model_input

# COMMAND ----------

import platform

PYTHON_VERSION = platform.python_version()

# COMMAND ----------

#Recommend packaging your environment with its dependencies and logging with the model
conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python={}".format(PYTHON_VERSION),
        "pip",
        {
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "pyspark=={}".format(pyspark.__version__)
            ],
        },
    ],
    "name": "pyspark_env",
}

# COMMAND ----------

df = spark.createDataFrame([(1, 0, 3)], ["a", "b", "c"])

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf custom_pyfunc

# COMMAND ----------

#Specify a path for the custom model to be saved in
mlflow_pyfunc_model_path = "custom_pyfunc"

#Save and log the custom model to MLflow
with mlflow.start_run(run_name="VectorAssembler") as run:
  mlflow.pyfunc.log_model(
      mlflow_pyfunc_model_path,
      python_model=ModelWrapper(),
      conda_env=conda_env
  )
  run_id = run.info.run_id

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model('runs:/{}/{}'.format(run_id, mlflow_pyfunc_model_path))
test_predictions = loaded_model.predict(df)
display(test_predictions)
