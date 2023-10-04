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
from mlflow.tracking.client import MlflowClient
import requests
import time

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
trainingDF = vecAssembler.transform(trainingDF)[[('features')]]
testingDF = vecAssembler.transform(testingDF)[[('features')]]
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
    assembled_vector = vecAssembler.transform(model_input)[[('features')]]
    
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
testingDF = testingDF.select([col(c).cast("long") for c in testingDF.columns])
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

model_name = "VectorAssembler_ExtendedIsolationForestModel"
model_serving_endpoint_name ="VectorAssembler_ExtendedIsolationForestModel"

def get_latest_model_version(model_name: str):
  client = MlflowClient()
  models = client.get_latest_versions(model_name, stages=["None"])
  new_model_version = models[0].version
  return new_model_version

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }

java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
instance = tags["browserHostName"]

# COMMAND ----------

my_json = {
  "name": model_serving_endpoint_name,
  "config": {
   "served_models": [{
     "model_name": model_name,
     "model_version": get_latest_model_version(model_name=model_name),
     "workload_size": "Small",
     "scale_to_zero_enabled": True
   }]
 }
}

# COMMAND ----------

def func_create_endpoint(model_serving_endpoint_name):

  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url = f"{endpoint_url}/{model_serving_endpoint_name}"
  r = requests.get(url, headers=headers)

  if "RESOURCE_DOES_NOT_EXIST" in r.text:

    print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
    re = requests.post(endpoint_url, headers=headers, json=my_json)

  else:

    new_model_version = (my_json['config'])['served_models'][0]['model_version']
    print("This endpoint existed previously! We are updating it to a new config with new model version: ", new_model_version)
  
    url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
    re = requests.put(url, headers=headers, json=my_json['config']) 

    import time,json

    url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
    retry = True
    total_wait = 0

    while retry:

      r = requests.get(url, headers=headers)

      assert r.status_code == 200, f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
      endpoint = json.loads(r.text)

      if "pending_config" in endpoint.keys():

        seconds = 10
        print("New config still pending")

        if total_wait < 6000:

          print(f"Wait for {seconds} seconds")
          print(f"Total waiting time so far: {total_wait} seconds")
          time.sleep(10)
          total_wait += seconds

        else:

          print(f"Stopping,  waited for {total_wait} seconds")
          retry = False  

      else:

        print("New config in place now!")
        retry = False

  assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"

def func_delete_model_serving_endpoint(model_serving_endpoint_name):

  endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
  url =  f"{endpoint_url}/{model_serving_endpoint_name}" 
  response = requests.delete(url, headers=headers)

  if response.status_code != 200:

    raise Exception(f"Request failed with status {response.status_code}, {response.text}")

  else:
    
    print(model_serving_endpoint_name, "endpoint is deleted!")

# COMMAND ----------

func_create_endpoint(model_serving_endpoint_name)

# COMMAND ----------

def wait_for_endpoint():

    endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"

    while True:

        url =  f"{endpoint_url}/{model_serving_endpoint_name}"
        response = requests.get(url, headers=headers)
        assert response.status_code == 200, f"Expected an HTTP 200 response, received {response.status_code}\n{response.text}"
 
        status = response.json().get("state", {}).get("ready", {})

        if status == "READY": print(status); print("-"*80); return

        else: print(f"Endpoint not ready ({status}), waiting 10 seconds"); time.sleep(10) # Wait 10 seconds
        
api_url = mlflow.utils.databricks_utils.get_webapp_url()
 
wait_for_endpoint()
 
time.sleep(5)

# COMMAND ----------

input_df = input_example.to_dict(orient="split")

payload_json = {"dataframe_split": input_df}

display(payload_json)

# COMMAND ----------

def score_model(data_json: dict):
    url =  f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations"
    response = requests.request(method="POST", headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()
  
print(score_model(payload_json))

# COMMAND ----------

# func_delete_model_serving_endpoint(model_serving_endpoint_name)
