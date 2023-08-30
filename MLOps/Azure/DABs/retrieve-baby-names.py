# Databricks notebook source
import requests

response = requests.get('http://health.data.ny.gov/api/views/myeu-hzra/rows.csv')
csvfile = response.content.decode('utf-8')
dbutils.fs.put("dbfs:/Volumes/andrewcooleycatalog/andrewcooleyschema/managedvolume/babynames.csv", csvfile, True)