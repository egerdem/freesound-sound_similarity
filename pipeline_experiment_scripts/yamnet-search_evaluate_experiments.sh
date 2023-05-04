#!/bin/bash

SCRIPT_DIR="$(pwd)/pipeline_scripts/"
export PATH="$SCRIPT_DIR:$PATH"

#############################################################################

DATA_DIR="$(pwd)/data"
DATASET_NAME="FSD50K.eval_audio"
MODEL_NAME="audioset-yamnet-1"

EMBED_DIR="$DATA_DIR/embeddings/$DATASET_NAME"

#############################################################################

for file in "$EMBED_DIR/"*; do # for each embedding dir
    f=$(basename -- "$file")   # get the basename=embed_name
    if [[ $f == "$MODEL_NAME-"* ]]; then # if the embed contains model-
        echo "======================================================================="
        echo $f
        SUFFIX="${f/$MODEL_NAME-/""}" # Strip model name to get the suffix
        yamnet-search_evaluate.sh $SUFFIX
    fi
done

#############################################################################