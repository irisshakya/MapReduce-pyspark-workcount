
"""
    Author: Iris Shakya
    course: Big Data Technology
    Project: 2, tf-idf w/ similarity search( mapreduce + pyspark)
    instructor: Prof. Lei Xei
    Reference: https://towardsdatascience.com/tf-idf-calculation-using-map-reduce-algorithm-in-pyspark-e89b5758e64c
"""


# %%
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
import string
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import gc
from pyspark.sql.functions import monotonically_increasing_id

sc = SparkContext(master = 'local')
spark = SparkSession.builder.appName('pySpark word-count').config('spark.some.config.option', 'some-value')             .getOrCreate()

# %%
data = spark.sparkContext.textFile("/Users/iris/project2_demo.txt")

# %% [markdown]
# import regex as re
# regex_st = r'\d{8}'
# 
# # doc = data.map(lambda x: x.split(regex_st))
# doc = data.flatMap(lambda x: x.split())
# doc1 = doc.map(lambda x: (x.split(regex_st)))

# %%
my_para = data.map(lambda x: tuple(x.split('\n')))

# %%
#split_list = my_para.map(lambda x: list(x[0].split()))
split_list = my_para.map(lambda x: tuple(x[0].split()))
#split_list2 = split_list.filter(lambda x: x[0] is not None)


doc_id = split_list.zipWithIndex().keyBy(lambda x: x[1])
doc_id = doc_id.map(lambda x: tuple((x[0], *x[1])))
doc_id = doc_id.map(lambda x: tuple((x[0], x[1])))


# %%
my_kv = doc_id.flatMap(lambda x: tuple( [ ((x[0], i),1) for i in x[1]]))

# %%
my_kv.take(5)

#rdd = sc.parallelize(["b", "a", "c"])
#sorted(rdd.map(lambda x: (x, 1)).collect())
#[('a', 1), ('b', 1), ('c', 1)]
#combined.map(lambda x: (x[0], *x[1]))


# %%
reduce = my_kv.reduceByKey(lambda x, y: x+y).collect()

# %%
reduce

# %%



