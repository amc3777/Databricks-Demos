# Databricks notebook source
# MAGIC %sh
# MAGIC wget -O cba.pdf https://imgix.cosmicjs.com/25da5eb0-15eb-11ee-b5b3-fbd321202bdf-Final-2023-NBA-Collective-Bargaining-Agreement-6-28-23.pdf

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install -U pypdf chromadb langchain transformers

# COMMAND ----------

from pypdf import PdfReader

reader = PdfReader("cba.pdf")
number_of_pages = len(reader.pages)
i = 24
text = ""

while i < number_of_pages:
  page = reader.pages[i]
  text += page.extract_text()
  i += 1

print(str(len(text)))

# COMMAND ----------



# COMMAND ----------


