DeepCoNN
===
The code implementation for the paper：  
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation. In WSDM. ACM, 425-434.

# Environments
  + python 3.8
  + pytorch 1.6

# Dataset
  You need to prepare the following documents:  
  1. dataset(`/data/music/Digital_Music_5.json.gz`)  
   Download from http://jmcauley.ucsd.edu/data/amazon (Choose Digital Music)
  2. word2vec(`/data/embedding/GoogleNews-vectors-negative300.bin`)  
   Download from https://code.google.com/archive/p/word2vec/  
   Or get it from BaiduYun: https://pan.baidu.com/s/1ddaATD0GAPb8z_K2dmnV7A 提取码：l7ko   

# Running
Preprocess origin dataset in json format to train.csv,valid.csv and test.csv.  
**Rewrite some necessary settings** in this file before running it. 
```
python preprocess.py
```

Train and evaluate the model:
```
python main.py
```
