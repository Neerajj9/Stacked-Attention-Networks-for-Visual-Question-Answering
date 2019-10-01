## Stacked-Attention-Networks-for-Visual-Question-Answring-
Implementation of the paper "Stacked Attention Networks for Image Question Answering" in Tensorflow

### Data 
Download the standard VQA COCO dataset from [here](https://visualqa.org/download.html)


### Preprocessing
Run :
``` python vqa_preprocessing_v2.py --download False --split 1```

The `vqa_preprocessing_v2.py` file has been adapted from this repository. It creates a Json `vqa_raw_train.json` and `vqa_raw_test.json`. It has the following structure : {question id, image path, question, ans }. Here questions and answers are in actual text format. 
Place the two files in a folder `Raw Data`.
