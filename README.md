# pytextclassification
文本分类器，提供多种文本分类和聚类算法，支持句子和文档级的文本分类任务，支持二分类、多分类、多标签分类、包括传统机器学习方法和常用的预训练的模型，开箱即用。python3 torch开发。
包括 lr，xgb，rf，textcnn，lstm，lstm+att，fasttext，bert，robert，ernie，T5等模型


##example：
都可以直接在对应的模型文件的main函数看到训练执行的demo，直接下拉最下面的main函数可看到
 
  预训练模型demo：
    https://github.com/JMDang/pytextclassification/blob/main/pytextclassification/transformers_classification.py
  
  深度模型非预训练demo：
    https://github.com/JMDang/pytextclassification/blob/main/pytextclassification/rnn_classifier.py

  传统机器学习模型demo：
    https://github.com/JMDang/pytextclassification/blob/main/pytextclassification/classical_classifier.py
