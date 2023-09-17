# Databricks notebook source
import py4j
import pandas
import pyarrow
import numpy
import pyspark
from pyspark.sql.functions import col
import mlflow
from pyspark.ml.feature import VectorAssembler
import h2o
from pysparkling import *
from pysparkling.ml import *

# COMMAND ----------

hc = H2OContext.getOrCreate()

# COMMAND ----------

h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")
sparkDF = hc.asSparkFrame(h2o_df)
[trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])

# COMMAND ----------

vecAssembler = VectorAssembler()
vecAssembler.setOutputCol("features")
vecAssembler.setInputCols(trainingDF.columns)
trainingDF = vecAssembler.transform(trainingDF)
testingDF = vecAssembler.transform(testingDF)
display(trainingDF)

# COMMAND ----------

algo = H2OExtendedIsolationForest(
                                  sampleSize=256,
                                  ntrees=100,
                                  seed=123,
                                  extensionLevel=8)

model = algo.fit(trainingDF)

# COMMAND ----------

h2o_model_path = "file:/exported_mojo"
model.write().overwrite().save(h2o_model_path)

# COMMAND ----------

model = H2OMOJOModel.createFromMojo("file:///exported_mojo/mojo_model")

# COMMAND ----------

output = model.transform(testingDF)

# COMMAND ----------

display(output.select("prediction").toPandas())

# COMMAND ----------

artifacts = {"h2o_model": "file:///exported_mojo/mojo_model"}

# COMMAND ----------

class ModelWrapper(mlflow.pyfunc.PythonModel):

  def load_context(self, context):

    self.path = str(context.artifacts["h2o_model"])

  def predict(self, context, model_input, params=None):

    from pyspark.sql import SparkSession
    import pysparkling
    import pysparkling.ml
    from pyspark.ml.feature import VectorAssembler

    spark = SparkSession.builder.master("local[*]").appName("VecAssembler").getOrCreate()

    hc = H2OContext.getOrCreate()

    h2o_model = H2OMOJOModel.createFromMojo(f"file://{self.path}")

    model_input = spark.createDataFrame(model_input)

    vecAssembler = VectorAssembler(outputCol="features")
    vecAssembler.setInputCols(model_input.columns)
    assembled_vector = vecAssembler.transform(model_input)
    
    output = h2o_model.transform(assembled_vector)
    output = output.select("prediction").toPandas()

    return output

# COMMAND ----------

import platform
import requests
import tabulate
import future

PYTHON_VERSION = platform.python_version()

# COMMAND ----------

conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python={}".format(PYTHON_VERSION),
        "pip",
        "h2o-py",
        "openjdk=8",
        {
            "pip": [
                "mlflow=={}".format(mlflow.__version__),
                "pyspark=={}".format(pyspark.__version__),
                "pandas=={}".format(pandas.__version__),
                "py4j=={}".format(py4j.__version__),
                "pyarrow=={}".format(pyarrow.__version__),
                "numpy=={}".format(numpy.__version__),
                "h2o_pysparkling_3.4",
                "requests=={}".format(requests.__version__),
                "tabulate=={}".format(tabulate.__version__),
                "future=={}".format(future.__version__),
            ],
        },
    ],
    "name": "pyspark_env",
}

# COMMAND ----------

h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")
sparkDF = hc.asSparkFrame(h2o_df)
[trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])
testingDF = testingDF.select([col(c).cast("double") for c in testingDF.columns])
input_example = testingDF.toPandas()[:5]
display(input_example)

# COMMAND ----------

#Save and log the custom model to MLflow
with mlflow.start_run(run_name="VectorAssembler_ExtendedIsolationForestModel") as run:
  mlflow.pyfunc.log_model(
      artifact_path="VectorAssembler_ExtendedIsolationForestModel",
      python_model=ModelWrapper(),
      artifacts=artifacts,
      conda_env=conda_env,
      registered_model_name="VectorAssembler_ExtendedIsolationForestModel",
      input_example=input_example
      )
  
  run_id = run.info.run_id

# COMMAND ----------

import py4j
import pandas
import pyarrow
import numpy
import pyspark
from pyspark.sql.functions import col
import mlflow
from pyspark.ml.feature import VectorAssembler
import h2o
from pysparkling import *
from pysparkling.ml import *

# COMMAND ----------

df = spark.read.option("header", True).csv("dbfs:/Users/andrew.cooley@databricks.com/prostate.csv")
df = df.select([col(c).cast("double") for c in df.columns])
input_example = df.toPandas()
display(input_example)

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model('runs:/2868be83fbae4e27b84a9c499ecb8945/VectorAssembler_ExtendedIsolationForestModel')
test_predictions = loaded_model.predict(input_example)
display(test_predictions)
