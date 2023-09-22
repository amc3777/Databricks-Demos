# Databricks notebook source

instance = dbutils.widgets.get("host")
 
print(f"{instance}")