#!/usr/bin/env bash

# DIR="work_dirs/r_rtmdet_hrsc/tiny-9x-hrsc-psc10/"
# NUM="20230404_135511"
DIR=$1
NUM=$2

FILE=$DIR$NUM"/vis_data/"$NUM".json"
KEYS="loss loss_cls loss_bbox"
OUT=$DIR$NUM".svg"

python tools/analysis_tools/analyze_logs.py plot_curve $FILE --keys $KEYS --legend $KEYS --out $OUT
