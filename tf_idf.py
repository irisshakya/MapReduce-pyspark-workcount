# %%
"""
    Author: Iris Shakya
    course: Big Data Technology
    Project: 2, tf-idf w/ similarity search( mapreduce + pyspark)
    instructor: Prof. Lei Xei
    References: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.zipWithIndex.html
                https://towardsdatascience.com/tf-idf-calculation-using-map-reduce-algorithm-in-pyspark-e89b5758e64c
"""
import math


# %%
from pyspark import SparkConf, SparkContext
import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import gc
from pyspark.sql.functions import monotonically_increasing_id

sc = SparkContext(master = 'local')
spark = SparkSession.builder.appName('pySpark word-count').config('spark.some.config.option', 'some-value')             .getOrCreate()

# %%
data = spark.sparkContext.textFile("/Users/iris/project2_demo_2.txt")

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

# unpacking 2 tuple into 1 singular tuple to compute
doc_id = doc_id.map(lambda x: tuple((x[0], *x[1])))

# re-packing the tuple
doc_id = doc_id.map(lambda x: tuple((x[0], x[1])))
doc_id.take(3)


# %%
# separate into key-value pairs
# ('doc_id', 'term') --> (( 'doc_id', 'token'), 1 )
kv_mapper = doc_id.flatMap(lambda x: tuple( [ ((x[0], i),1) for i in x[1]]))

# %%
kv_mapper.take(5)

# %%
# do the term-count
# (('doc_id', 'token'), 1) --> (('doc_id', 'token'), [1++])
reduce_1 = kv_mapper.reduceByKey(lambda x, y: x+y)


#N = kv_mapper.map(lambda x: x[1]).sum()
reduce_1.take(4)

# %%

# tf = reducer_2.map(lambda x: (x[0], (len(doc_id1)/x[1])))
# tf.take(5)

# %%
# mapping for IDF start
# (('doc_id', 'token'), tf) --> ('token', ('doc_id', tf))


# tf = reducer_2.map(lambda x: (x[0], (len(doc_id1)/x[1])))

shuffle = reduce_1.map(lambda x: (x[0][1], (x[0][0], x[1])))
shuffle.take(3)

# %%
# tokenise for idf
# (('doc_id', 'token'), tf) --> ('token', ('doc_id', tf, 1))
mapper_2 = reduce_1.map(lambda x: (x[0][1], (x[0][0], x[1],1)))
mapper_2.take(2)

# %%
# only take term and token
# ('token', ('doc_id', tf,1)) --> ('token', 1)
mapper_3 = mapper_2.map(lambda x: (x[0], x[1][2]))
mapper_3.take(3)

# %%
# counting by token
# ('token', 1) --> ('token', [1++])
reducer_2 = mapper_3.reduceByKey(lambda x, y: x+y)
reducer_2.take(3)

# %%
# convert into python list so we can pass it into a map transformation
doc_id1 = doc_id.map(list).collect()


# %%
tf = reducer_2.map(lambda x: (x[0], (len(doc_id1)/x[1])))
tf.take(5)

# %%
idf = tf.map(lambda x: (x[0], math.log10(len(doc_id1)/x[1])))
idf.take(3)

# %%
joint_my = tf.join(idf)
joint_my.take(4)

# %%
# parallelize so we can convert list back into a RDD
# i dont know why this didnt work this time
#rdd = sc.parallelize(joint_my)

# %%
# ('doc_id', 'everything_else') --> (('doc_id'), ('token', 'tf', 'idf', 'tf-idf'))
# tf_idf = joint_my.map(lambda x: (x[1][0][0], ( x[0], x[1][0][1], x[1][1], x[1][0][1] * x[1][1])) ).sortByKey()
tf_idf = joint_my.map(lambda x: (x[0], ( x[1][0] * x[1][1])) )
tf_idf.collect()

# %%
prefix1 = 'dis_breast_cancer_dis'
prefix2 = 'gene_egfr+_gene'
doc_1 = tf_idf.filter(lambda x: x[1][0] == prefix1).map(lambda y: (y[0], y[1][3]))
doc_2 = tf_idf.filter(lambda x: x[1][0] == prefix2).map(lambda y: (y[0], y[1][3]))

# %%
doc_1.take(5)

# %%
doc_2.take(4)

# %%



