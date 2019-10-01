## Stacked-Attention-Networks-for-Visual-Question-Answring-
Implementation of the paper "Stacked Attention Networks for Image Question Answering" in Tensorflow

### Data 
Download the standard VQA COCO dataset from [here](https://visualqa.org/download.html)


### Preprocessing
Run :
``` python vqa_preprocessing_v2.py --download False --split 1```

The `vqa_preprocessing_v2.py` file has been adapted from this repository. It creates a Json `vqa_raw_train.json` and `vqa_raw_test.json`. It has the following structure : {question id, image path, question, ans }. Here questions and answers are in actual text format. 
Place the two files in a folder `Raw Data`.

Run : 
```python save_QA.py```

The `save_QA.py` file takes input the json created earlier and converts the text data into word embeddings using Google Word Vectors. It then saves the Word embeddings of the question, Word embeddings of the question of the answer and the question id in the form of `.h5` for each data point in a separate file. All files are stored in `Final_Data/VQA` folder.
