# Databricks notebook source
import mlflow
import h2o
from pysparkling import *
from pysparkling.ml import H2OExtendedIsolationForest
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

# COMMAND ----------

hc = H2OContext.getOrCreate()

# COMMAND ----------

h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")
sparkDF = hc.asSparkFrame(h2o_df)

# COMMAND ----------

with mlflow.start_run() as run:

  vecAssembler = VectorAssembler()
  vecAssembler.setOutputCol("features")
  vecAssembler.setInputCols(sparkDF.columns)

  algo = H2OExtendedIsolationForest(
                                  sampleSize=256,
                                  ntrees=100,
                                  seed=123,
                                  extensionLevel=len(sparkDF.columns) - 1)

  pipeline = Pipeline(stages=[vecAssembler, algo])

  [trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])
  model = pipeline.fit(trainingDF)

  mlflow.spark.log_model(model, artifact_path="spark-ml-pipeline", registered_model_name="VecAssembler_H2OExtendedIsolationForest")

mlflow.end_run()

# COMMAND ----------

logged_model = f'runs:/{run.info.run_id}/spark-ml-pipeline'

loaded_model = mlflow.spark.load_model(logged_model)

prediction = loaded_model.transform(testingDF)

display(prediction)

# COMMAND ----------

h2o_df = h2o.import_file("https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv")
sparkDF = hc.asSparkFrame(h2o_df)
[trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])
testingDF.write.mode("overwrite").saveAsTable("andrewcooleycatalog.andrewcooleyschema.prostate_anom_testing")
