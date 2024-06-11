For Koalas it's unclear that I run it correctly. I use this:
```python
ks.set_option('compute.default_index_type', 'distributed')
```
It improves performance. I also tried various Spark settings for number
of cores and memory, e.g.,:
```python
from pyspark import SparkConf, SparkContext
conf = SparkConf()
conf.get('spark.executor.cores', '40')
conf.get('spark.executor.memory', '160g')
SparkContext(conf=conf)
```

Still, nothing improved performance and the memory failures were still there.