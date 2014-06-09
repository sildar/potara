#!/bin/bash

if [ -n "$1" ]
then
    infolder=$1
else
    echo "The input dir is mandatory"
    exit
fi

DIRS=`ls -d $infolder*/`
NEWCLUSTERDIR="./clusters04"
NEWSUMMARYDIR="./summaries04"

rm $NEWSUMMARYDIR/D3*
rm $NEWCLUSTERDIR/d3*

mkdir "$NEWCLUSTERDIR"
mkdir "$NEWSUMMARYDIR"

for DIR in $DIRS
do
    DIRNAME=`basename $DIR`
    ./cluster_sentences.py $DIR $NEWCLUSTERDIR/$DIRNAME
    outname="${DIRNAME:0:6}.M.100.T.111"
    # uppercase
    outname=${outname^^}
    ./fusion.py $NEWCLUSTERDIR/$DIRNAME $NEWSUMMARYDIR/$outname
done



