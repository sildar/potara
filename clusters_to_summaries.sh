#!/bin/bash

if [ -n "$1" ]
then
    infolder=$1
else
    echo "The input dir is mandatory"
    exit
fi

NEWSUMDIR="./ilpSumm04"

FILES=`ls $infolder`

mkdir "$NEWSUMDIR"

for FILE in $FILES
do
    echo "$FILE"
    ./ilp_summary.py $infolder/$FILE $NEWSUMDIR/$FILE
done


# copy and eval
cp $NEWSUMDIR/* ./res_duc_04/gillick_stem/eval/peers/2/
cd ./res_duc_04/gillick_stem/
./ROUGE-1.5.5.pl -e ./data/ -n 4 -2 -4 -u -m -a -l 100 -x -c 95 -r 1000 -f A -p 0.5 -t 0 t2.rouge.in > t2.rouge.out
cat t2.rouge.out | grep "111 ROUGE"

