# Databricks notebook source
import mlflow
from mlflow.tracking.client import MlflowClient
import requests

CATALOG = "andrewcooleycatalog"
SCHEMA = "andrewcooleyschema"

mlflow.set_registry_uri('databricks-uc')

model_name = f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_regressor'
model_serving_endpoint_name ='airbnb_price_prediction_regressor_endpoint'

def get_latest_model_version(model_name: str):
  client = MlflowClient()
  alias_mv = client.get_model_version_by_alias(model_name, "champion")
  return alias_mv.version
 

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
 
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
  }
 
java_tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
tags = sc._jvm.scala.collection.JavaConversions.mapAsJavaMap(java_tags)
instance = tags["browserHostName"]
 
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

func_create_endpoint(model_serving_endpoint_name)
