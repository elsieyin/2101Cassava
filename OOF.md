2021/1/23 Frank：

k-fold cross-validation是一种统计方法里面的采样技术，基本步骤如下
打乱数据集然后分为k fold，每一个fold作为测试集，其余fold作为训练集，依次执行。
其中每一fold预测出来的结果称为out-of-fold prediction.  从字面啊意思很容易理解，不是这个fold，其余作为训练集的fold产生的结果

out-of-fold prediction有两个作用，一个是用作评估模型性能，一个是用作融合

评估模型性能有两种方法
方法一：每一个fold预测出来的out-of-predictions求出来的结果做mean
方法二：将所有fold的out-of-prediction做aggregate，这加起来就是整个数据集的预测结果，然后将这个预测跟语气结果作比较，计算得分。
这两种方法都是合理的，做好是用mean

out-fo-fold 用作融合ensemble，ensemble是在相同训练集上用多个模型进行预测

最出名的工作是在小麦检测上用的OOF，从那时开始，几乎机器学习竞赛OOF就是必备的

如何最佳的组合和校正预测是关键

定义两个概念 开始是用的模型称为 基本模型(base-model) 比如efficientNet  其他模型的预测成为 元模型(meta-model)比如resnet 然后与base-model的预测最佳结合。也成为stacking。

百度没这方面的资料，这是国外一个phd写的参考：https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/#:%7E:text=An%20out%2Dof%2Dfold%20prediction,example%20in%20the%20training%20dataset.

基本上的意思就是我写的那些中文. 结合英文看看