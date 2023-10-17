# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS andrewcooleycatalog;
# MAGIC USE CATALOG andrewcooleycatalog;
# MAGIC CREATE SCHEMA IF NOT EXISTS andrewcooleyschema;
# MAGIC USE SCHEMA andrewcooleyschema;
# MAGIC CREATE VOLUME IF NOT EXISTS managedvolume;