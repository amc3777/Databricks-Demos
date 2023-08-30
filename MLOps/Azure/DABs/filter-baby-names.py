# Databricks notebook source
babynames = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/Volumes/andrewcooleycatalog/andrewcooleyschema/managedvolume/babynames.csv")
babynames.createOrReplaceTempView("babynames_table")
years = spark.sql("select distinct(Year) from babynames_table").rdd.map(lambda row : row[0]).collect()
years.sort()
dbutils.widgets.dropdown("year", "2014", [str(x) for x in years])
display(babynames.filter(babynames.Year == dbutils.widgets.get("year")))