# PaddlePaddle-CNN
用Paddle搭建一个简单的文本CNN网络

CNN结构：Embedding-Conv1d-Maxpool-Flatten-Dense

在构建卷积层的时候，我们可以用飞桨自带的sequence_conv进行一维卷积。但是这里我采用了飞桨的Conv2D二维卷积来近似文本的一维卷积（这个思路来自于tensorflow的一维卷积）。由于在嵌入层之后的一维卷积需要卷积来自维度的信息，所以用sequence_conv需要一些技巧，而用Conv2D就灵活地多。为了进行文本卷积，我们可以假设二维卷积的输入只有一个通道，同时二维输入的高度为嵌入维度，宽度为文本长度。而卷积核在高度方向大小为嵌入维度，卷积步长stride在高度方向大小为1（其实为任意都可，在这里卷积核在高度方向只行进一次，因为卷积核在这里的大小等于嵌入维度）。

#### cnn_dygraph_pd: 基于动态图模式的CNN
#### cnn_pd: 基于静态图模式的CNN
#### cnn_pd_pre_trained: 预训练词向量，训练阶段词向量不参与训练，词向量通过lookup模块进行查询。
#### data_utils: 读取文本文件模块

### lookup模块
集成了PreprocessVector和EnVectorizer两个类。PreprocessVector将需要查询的词向量字典按照字母分成多个小份文件进行并行查询，提高速度。具体参数可以参考类内的注释。EnVectorizer用来查询字典中的词向量。
### data_utils
用load_data读取文件，返回编码后的文本集，one hot后的文本类别标记，文本的词典，文本的所有按照标记排列的词，以及测试集大小。文件地址在load_data_and_label里修改。
