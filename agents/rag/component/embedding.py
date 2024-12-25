import numpy as np
from transformers import AutoModel
from zhipuai import ZhipuAI
import os
from typing import List


class Zhipuembedding:

    def __init__(self, path: str = ' ', api_key="c59db5e044cd9cd453a49b462a659697.RD2fEoEAwM5EhPuE"):
        # lyq
        client = ZhipuAI(api_key=api_key)
        self.embedding_model = client
        print("embedding模型初始化")

    def get_embedding(self, content: str = ' '):
        response = self.embedding_model.embeddings.create(
            model="embedding-2",
            input=content
        )
        return response.data[0].embedding

    def compare_v(cls, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def compare(self, text1: str, text2: str):
        embed1 = self.embedding_model.embeddings.create(
            model="embedding-2",
            input=text1
        ).data[0].embedding

        embed2 = self.embedding_model.embeddings.create(
            model="embedding-2",
            input=text2
        ).data[0].embedding

        return np.dot(embed1, embed2) / (np.linalg(embed1) * np.linalg.norm(embed2))
