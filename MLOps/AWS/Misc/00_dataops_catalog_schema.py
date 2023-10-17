# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS andrewcooleycatalog;
# MAGIC USE CATALOG andrewcooleycatalog;
# MAGIC CREATE SCHEMA IF NOT EXISTS airbnb_data;
# MAGIC CREATE SCHEMA IF NOT EXISTS monitoring_example;
# MAGIC USE SCHEMA airbnb_data;
# MAGIC CREATE VOLUME IF NOT EXISTS unstructured_data;
