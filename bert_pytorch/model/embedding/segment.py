import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        """ 3 表示有 3 种片段类型（0：padding, 1:sent_A, 2:sent_B） """
        '''参与训练: SegmentEmbedding 的权重矩阵会参与训练，除了 padding_idx=0 对应的嵌入向量。
不更新的部分: padding_idx=0 的嵌入向量会被固定为全零，并且不会参与梯度更新。'''
        super().__init__(3, embed_size, padding_idx=0)