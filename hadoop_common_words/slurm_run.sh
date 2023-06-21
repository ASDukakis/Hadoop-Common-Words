#!/bin/bash
#DO NOT MODIFIY!!!
#SBATCH --job-name=ASSIGN_1
#SBATCH -o ASSIGN_1.out

echo "Java path $JAVA_HOME"
echo "Hadoop path $HADOOP_HOME"

k=10

echo "Compiling assignment 1"
hadoop com.sun.tools.javac.Main TopkCommonWords.java
jar cf cm.jar TopkCommonWords*.class
echo "Deleting previous files"
hdfs dfs -rm -r ./temp/
hdfs dfs -mkdir -p ./temp/
hdfs dfs -mkdir ./temp/input/
echo "Uploading input files"
hdfs dfs -put ./data/* ./temp/input/
echo "Submit job"
hadoop jar cm.jar TopkCommonWords ./temp/input/task1-input1.txt ./temp/input/task1-input2.txt ./temp/input/stopwords.txt ./temp/output/ $k
echo "Job finished. Print Top $k words."
hdfs dfs -cat ./temp/output/part-r-00000
hdfs dfs -get ./temp/output/part-r-00000 . && mv part-r-00000 ./output.txt
python3 ./check_common.py ./answer.txt ./output.txt

