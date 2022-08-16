#!/bin/bash

cd ./change_by_model
python eval-tI.py

cd ./results
python posteval.py
python extractterm_pred.py

cd ..
cd ..
cd ./term_classification_training
python predict.py --predict_text_file ../change_by_model/results/BC5-sents --predict_match_file ../change_by_model/results/BC5-tags
cp ./output/lr5e-05_mask0.15min-word-count-2/BC5-sents_match.txt ../change_by_model/results

cd ..
cd ./change_by_model/results
python addtype.py
python change_type.py