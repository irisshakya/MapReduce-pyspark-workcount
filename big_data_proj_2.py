"""
    project 2, TF-IDF; cosine similarity 
    author: Iris Shakya
    course: Big Data Tech
    
    references: 
            https://dzone.com/articles/calculating-tf-idf-with-apache-spark
            https://datascience-enthusiast.com/Python/text_analysis.html
            https://notebook.community/MingChen0919/learning-apache-spark/notebooks/04-miscellaneous/TF-IDF
"""


from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
import string
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import gc



from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


sc = SparkContext(master = 'local')
spark = SparkSession.builder.appName('pySpark word-count')             .config('spark.some.config.option', 'some-value')             .getOrCreate()


data = spark.sparkContext.textFile("/Users/iris/data.txt")


# SEPARATE EACH NEWLINE AS A list
my_para = data.flatMap(lambda x: x.split('\n'))


my_list = my_para.map(lambda x: list(x.split(" ")))



from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql import functions as F


row = Row("doc") # Or some other column name
my_df = my_list.flatMap(row).toDF(schema=StringType())
my_df.show(truncate=True)


df = my_df.withColumn('doc_id', F.monotonically_increasing_id())
num_doc = df.count()



from pyspark.sql.functions import col, regexp_replace, split

df_trans = df.withColumn("value",
split(regexp_replace(col("value"), r"(^\[)|(\]$)|(')", ""), ", "))

#df_trans.printSchema()

#new_df = df_trans.withColumn("token", F.explode("value").alias('token')).show(truncate=True)
new_df = df_trans.select('*', F.explode('value').alias('token'))

new_df.show()



#CAlCULATE TF
TF = new_df.groupBy("doc_id").agg(F.count("token").alias("doc_len"))             .join(new_df.groupBy("doc_id", "token")             .agg(F.count("value").alias("word_count")), ['doc_id']) 

#print(new_df.columns)



TF = TF.withColumn("tf", F.col("word_count") / F.col("doc_len"))       .drop("doc_len", "word_count")


new_df.show()
TF.show()



# Calculate IDF
IDF = new_df.groupBy("token").agg(F.countDistinct("doc_id").alias("df"))
IDF = IDF.select('*', (F.log(num_doc / (IDF['df'] + 1))).alias('idf'))

# Calculate TF-IDF
TF_IDF = TF.join(IDF, ['token']).withColumn('tf-idf', F.col('tf') * F.col('idf'))



TF_IDF.show()

