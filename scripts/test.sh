#!/bin/bash

DATASET="synthetic_16_32_64"
METHODS="simpleKM canopyKM"
INITS="covertree random firstk kmeanspp"
NUM_CLUSTERS=16
NUM_ITER=100

DIR_NAME="tmp_test/"
mkdir -p $DIR_NAME

python scripts/generateData.py 16 32 64 4

dist/cover_tree data/synthetic_16_32_64-0.dat data/synthetic_16_32_64-test.dat

for METHOD in $METHODS
do
for INIT in $INITS
do
	#run
	#valgrind --leak-check=yes
	echo $METHOD
        echo $INIT
	dist/k_means --method "$METHOD" --num-clusters "$NUM_CLUSTERS" --init-type "$INIT"  --num-iterations $NUM_ITER --output-state-interval 1000 --output-model $DIR_NAME --dataset data/"$DATASET"
        echo "==========================================="
        echo "==========================================="
done
done
rm -rf tmp_test
echo 'done'

# dist/k_means --method simpleKM --num-clusters 16 --init-type random  --num-iterations 10 --output-state-interval 20 --output-model out/ --dataset data/synthetic_16_32_64
