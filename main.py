# 1. 导入必要的库
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 2. 加载并预处理 IMDB 数据集
# 设置参数
max_features = 10000  # 词汇表大小（只考虑最常见的10000个词）
maxlen = 500         # 每条评论的最大长度（超过则截断，不足则填充）
batch_size = 32

# 加载数据集
print("正在加载数据...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)#keras里面已经是“嵌入数字”的版本 原始在
print(f"训练序列数量: {len(x_train)}")
print(f"测试序列数量: {len(x_test)}")

# 序列填充 (Padding)
# SimpleRNN 要求输入序列长度是固定的
print("正在填充序列（将样本处理成相同长度）...")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print(f"x_train 的形状: {x_train.shape}")
print(f"x_train[0]是: {x_train[0]}")
print(f"x_test 的形状: {x_test.shape}")


# 3. 构建简单的 RNN 模型

# 嵌入层 (Embedding Layer)
# 将正整数（单词索引）转换为固定大小的密集向量
# 10000: 词汇表大小, 32: 嵌入向量的维度
#经过嵌入向量变成(25000, 500, 32) 原来是500 个由数字组成的序列，现在变成了 500 个 32 维向量的序列。模型现在处理的不再是数字列表，而是包含丰富语义信息的向量矩阵。
#一个 32 维向量就是一个包含 **32 个有序浮点数（小数）**的列表或数组。  我们无法确切知道每个维度代表什么，但可以推测它们编码了情感信息
#核心结论：向量的几何意义这 32 个数字的真正意义在于它们的相对关系：距离 (Distance)： 向量空间中距离越近的词，语义越相似（例如 "great" 和 "fantastic"）。方向 (Direction)： 向量的相加和相减可以捕捉关系。著名的例子是：国王-男人+女人≈女王 32 维向量的作用就是让模型能够理解并利用这种复杂的语义关系进行高效的文本分析。
model = Sequential()
model.add(Embedding(max_features, 32)) #一开始这些32个参数的向量是随机初始化的 学习后 慢慢适用于网络了 每个相同的单词在一个epoch里面向量是一样的  下一个epoch会微调（整的方向是最小化最终的损失）

# SimpleRNN 层
# 32: 输出空间的维度（隐藏单元的数量）
model.add(SimpleRNN(10))

# 输出层 (Dense Layer)
# 1: 输出一个单一的数值
# 'sigmoid': 激活函数，将输出压缩到 0-1 之间，非常适合二分类问题
model.add(Dense(1, activation='sigmoid'))


# 4. 编译模型
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', # 二分类问题的标准损失函数
              metrics=['accuracy'])

# 打印模型结构
model.summary()


# 5. 训练模型
print("正在训练模型...")
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2) # 在训练过程中使用20%的数据进行验证


# 6. 评估模型
# 在测试集上评估模型的性能
score = model.evaluate(x_test, y_test, verbose=0)
print("\n评估完成！")
print(f'测试集损失: {score[0]}')
print(f'测试集准确率: {score[1]}')

# 7. 进行一个简单的预测
# 我们可以用一个样本来测试
# (注意：真实应用中需要将新文本转换为数字序列)
sample_prediction = model.predict(x_test[:1])
print(f"\n对第一个测试样本的预测原始输出: {sample_prediction[0][0]}")
if sample_prediction[0][0] > 0.5:
    print("预测结果: 正面评论")
else:
    print("预测结果: 负面评论")

print(f"实际标签: {'正面评论' if y_test[0] == 1 else '负面评论'}")