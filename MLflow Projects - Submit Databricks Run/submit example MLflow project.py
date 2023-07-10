# Databricks notebook source
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

import mlflow

project_uri = "https://github.com/amc3777/Practical-Deep-Learning-at-Scale-with-MLFlow#chapter05"
backend = "databricks"
backend_config="cluster_spec.json"
parameters={"pipeline_steps":"all"}

mlflow.projects.run(uri=project_uri, parameters=parameters, backend=backend, backend_config=backend_config)
