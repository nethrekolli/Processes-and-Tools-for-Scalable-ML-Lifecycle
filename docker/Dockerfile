FROM java:openjdk-8-jdk

ENV spark_ver 3.0.1
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Get Spark and Anaconda.
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh && \
	conda install python=3.6 && \
	pip install tensorflow==1.15.0 && \
	pip install sparkflow && \
	mkdir -p /opt && \
    cd /opt && \
    curl https://downloads.apache.org/spark/spark-${spark_ver}/spark-${spark_ver}-bin-hadoop2.7.tgz | \
        tar -zx && \
    ln -s spark-${spark_ver}-bin-hadoop2.7 spark && \
    echo Spark ${spark_ver} installed in /opt

ADD start-common.sh start-worker.sh start-master.sh /
RUN chmod 777 /start-common.sh /start-master.sh /start-worker.sh && \
	sed -i -e 's/\r$//' start-common.sh && \
	sed -i -e 's/\r$//' start-master.sh && \
	sed -i -e 's/\r$//' start-worker.sh
ENV PATH $PATH:/opt/spark/bin