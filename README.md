# Emotion-analysis-of-Chinese-text-based-on-machine-learning
文件介绍：
Start.py：用来启动系统界面，实行人机交互；
Word2vec_train.py：对使用到word2vec词向量生成方法的模型进行训练；
Embedding_train.py：对使用到embedding词向量生成方法的模型进行训练；
Stop_words.txt：停用词表；

文件夹介绍：
model：存放了神经网络的模型代码
--em_textcnn.py：针对使用到embedding词向量生成方法的模型而写
--textcnn.py：针对使用到word2vec词向量生成方法的模型而写
solver：存放了对数据的一些操作代码
--cut.py：使用不同的分词手段对文本进行分词
--dataloader.py：生成自己的数据集的类
--predict.py：对文本进行预测的代码函数
保存模型：存放了训练好的各个模型
--.pt：以pt结尾的为神经网络的模型
--.model：以model结尾的为word2vec的词向量模型
数据集：存放了原始的数据集，以及使用不同分词方法分词后的数据集
系统界面：存放了绘制好的界面ui文件以及对应的py文件
--untitled.ui：主界面的ui文件
-- untitled1.ui：子界面的ui文件
--srs.py：主界面的代码文件
-- srs1.py：子界面的代码文件
