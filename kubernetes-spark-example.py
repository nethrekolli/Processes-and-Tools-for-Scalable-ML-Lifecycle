from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from sparkflow.tensorflow_async import SparkAsyncDL
from pyspark.sql.functions import rand
from sparkflow.graph_utils import build_graph
from sparkflow.graph_utils import build_adam_config
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
import tensorflow as tf


def small_model():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
    layer1 = tf.layers.dense(x, 256, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
    layer2 = tf.layers.dense(layer1, 256, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
    out = tf.layers.dense(layer2, 10, kernel_initializer=tf.glorot_uniform_initializer())
    z = tf.argmax(out, 1, name='out')
    loss = tf.losses.softmax_cross_entropy(y, out)
    return loss


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("kubernetes-spark-project") \
        .config("spark.jars", "/opt/spark/jars/gcs-connector-latest-hadoop2.jar") \
        .getOrCreate()

    # spark._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile","key.json")

    # Read in mnist_train.csv dataset
    df = spark.read.option("inferSchema", "true").csv('gs://kubernetes-spark-project-bucket/mnist_train_small.csv').orderBy(rand())

    # Build the tensorflow graph
    mg = build_graph(small_model)

    # Build the adam optimizer
    adam_config = build_adam_config(learning_rate=0.001, beta1=0.9, beta2=0.999)

    # Setup features
    vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')
    encoder = OneHotEncoder(inputCol='_c0', outputCol='labels', dropLast=False)

    spark_model = SparkAsyncDL(
        inputCol='features',
        tensorflowGraph=mg,
        tfInput='x:0',
        tfLabel='y:0',
        tfOutput='out:0',
        tfOptimizer='adam',
        miniBatchSize=300,
        miniStochasticIters=1,
        shufflePerIter=True,
        iters=50,
        predictionCol='predicted',
        labelCol='labels',
        partitions=4,
        verbose=1,
        optimizerOptions=adam_config
    )

    # Create and save the Pipeline
    p = Pipeline(stages=[vector_assembler, encoder, spark_model]).fit(df)

    test_df = spark.read.option("inferSchema", "true").csv('gs://kubernetes-spark-project-bucket/mnist_test.csv').orderBy(rand())

    # Run predictions and evaluation
    predictions = p.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(
    labelCol="_c0", predictionCol="predicted", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (accuracy))