#!/bin/bash

export CLASSPATH='/home/u1138972/final_submission/stanford-ner-2017-06-09/stanford-ner.jar'
export STANFORD_MODELS='/home/u1138972/final_submission/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz'

pip install --user virtualenv

python2.7 -m virtualenv sample_env2

source sample_env2/bin/activate

pip install numpy
python -m pip install fuzzywuzzy
python -m pip install scipy
python -m pip install scikit-learn
python -m pip install nltk
python -m pip install spacy
python -m pip install nameparser
python -m spacy download en
python -m nltk.downloader 'punkt'
python -m nltk.downloader 'popular'

unzip stanford-corenlp-full-2017-06-09.zip
pip install stanfordcorenlp
# unzip stanford-ner-2017-06-09.zip 

python sample_new.py $1

