#!/bin/bash

DATASETS="mnist8m" #"synthetic_3_2_5"
METHODS="simpleKM canopyKM"
INITS="covertree random covertree firstk kmeanspp"
NUM_CLUSTERS="256"
NUM_ITER=100

for REP in `seq 0 5`
do
for NUM_CLUSTER in $NUM_CLUSTERS
do 
for DATASET in $DATASETS
do
for METHOD in $METHODS
do
for INIT in $INITS
do
    #Create the directory structure
    time_stamp=`date "+%b_%d_%Y_%H.%M.%S"`
    DIR_NAME='out'_$REP/$DATASET/$METHOD/$NUM_CLUSTER/$INIT/
    mkdir -p $DIR_NAME

    #save details about experiments in an about file
    echo Running FMM inference using $METHOD | tee -a $DIR_NAME/log.txt
    echo initializing with $INIT | tee -a $DIR_NAME/log.txt
    echo For dataset $DATASET | tee -a $DIR_NAME/log.txt
    echo For number of iterations $NUM_ITER | tee -a $DIR_NAME/log.txt
    echo For number of clusters $NUM_CLUSTER | tee -a $DIR_NAME/log.txt
    echo with results being stored in $DIR_NAME
    echo Using $NT threads on bros | tee -a $DIR_NAME/log.txt

    #run
    #valgrind --leak-check=yes
    echo $METHOD
    mpirun -v --mca btl_tcp_if_exclude lo,eth0 -np 4 --report-bindings -machinefile config/hosts dist/k_means --method "$METHOD" --num-clusters "$NUM_CLUSTER" --init-type "$INIT"  --num-iterations $NUM_ITER --output-state-interval 10 --output-model $DIR_NAME --dataset data/"$DATASET" | tee -a $DIR_NAME/log.txt
        echo "==========================================="
        echo "==========================================="
done
done
done
done
done
echo 'done'

# dist/k_means --method simpleKM --num-clusters 10 --init-type random  --num-iterations 10 --output-state-interval 10 --output-model out/ --dataset data/mnist8m
