from tqdm import tqdm
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./rag/component")))
from embedding import Zhipuembedding
import os
import json
from typing import List
from data_chunker import ReadFile
from sklearn.cluster import KMeans


class Vectordatabase:

    # 初始化方法，传入一个字块列表
    def __init__(self, docs: List = []) -> None:
        self.docs = docs
        self.entire_vectors = []
        self.topic_vectors = []
        self.document = []

    # 对字块列表进行，批量的embedded编码，传入embedding模型，返回一个向量列表
    def get_topic_vector(self, EmbeddingModel) -> List[List[float]]:
        self.topic_vectors = []
        for doc in tqdm(self.docs):
            post_index = doc.find("帖子")
            content_before_post = doc[:post_index]
            self.topic_vectors.append(EmbeddingModel.get_embedding(content_before_post, 3))
            # self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.topic_vectors

    def get_entire_vector(self, EmbeddingModel) -> List[List[float]]:
        self.entire_vectors = []
        for doc in tqdm(self.docs):
            self.entire_vectors.append(EmbeddingModel.get_embedding(doc), 3)
            # self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.entire_vectors

    # 把向量列表存储到json文件中，把子块列表也存储到json文件,默认路径为'database'
    def persist(self, path: str = 'database') -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        with open(f"{path}/entire_vectors.json", 'w', encoding='utf-8') as f:
            json.dump(self.entire_vectors, f)
        with open(f"{path}/topic_vectors.json", 'w', encoding='utf-8') as f:
            json.dump(self.topic_vectors, f)

    # 加载json文件中的向量和字块，得到向量列表、字块列表,默认路径为'database'
    def load_vector(self, path: str = 'database', load_ratio=1) -> None:
        with open(f"{path}/topic_vectors.json", 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            limit = int(len(all_data) * load_ratio)
            self.topic_vectors = all_data[:limit]
            print("总共有" + str(len(self.topic_vectors)) + "条数据")
        with open(f"{path}/entire_vectors.json", 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            limit = int(len(all_data) * load_ratio)
            self.entire_vectors = all_data[:limit]
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            limit = int(len(all_data) * load_ratio)
            self.document = all_data[:limit]

    # 求向量的余弦相似度，传入两个向量和一个embedding模型，返回一个相似度
    def get_similarity(self, vector1: List[float], vector2: List[float], embedding_model) -> float:
        return embedding_model.compare_v(vector1, vector2)

    def get_similarity_both(self, pos_query: List[float], neg_query: List[float], data: List[float],
                            embedding_model) -> float:
        pos_similarity = embedding_model.compare_v(pos_query, data)
        neg_similarity = embedding_model.compare_v(neg_query, data)
        return pos_similarity - neg_similarity

    def query_both(self, pos_query: str, neg_query: str, EmbeddingModel, k: int = 1, n: int = 3,
                   multi: bool = True, invisible: int = -1) -> \
            List[str]:
        visible_topic_vectors = [vector for idx, vector in enumerate(self.topic_vectors) if idx != invisible]
        visible_entire_vectors = [vector for idx, vector in enumerate(self.entire_vectors) if idx != invisible]
        visible_document = [document for idx, document in enumerate(self.document) if idx != invisible]
        if neg_query:
            pos_query_vector = EmbeddingModel.get_embedding(pos_query)
            neg_query_vector = EmbeddingModel.get_embedding(neg_query)
            result = np.array([self.get_similarity_both(pos_query_vector, neg_query_vector, vector, EmbeddingModel)
                               for vector in visible_topic_vectors])
        else:
            pos_query_vector = EmbeddingModel.get_embedding(pos_query)
            result = np.array([self.get_similarity(pos_query_vector, vector, EmbeddingModel)
                               for vector in visible_topic_vectors])

        if k == 1 or not multi:
            return np.array(visible_document)[result.argsort()[-k:][::-1]].tolist()
        else:
            top_nk_indices = result.argsort()[-n * k:][::-1]
            selected_vectors = np.array(visible_entire_vectors)[top_nk_indices].tolist()
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(selected_vectors)
            labels = kmeans.labels_
            cluster_min_indices = []
            for cluster_id in np.unique(labels):
                cluster_indices = np.where(labels == cluster_id)[0]
                origin_indices = [top_nk_indices[i] for i in cluster_indices]
                min_index = min(origin_indices)
                cluster_min_indices.append(min_index)
            return np.array(visible_document)[cluster_min_indices].tolist()


if __name__ == "__main__":
    cur_database = 'xihu'
    database_path = '../databases/' + cur_database
    rag_data_path = "/home/wangb/cyo/graduation/output/" + cur_database + '/'
    data_loader = ReadFile(rag_data_path)
    docs = data_loader.get_all_chunk_content()
    database = Vectordatabase(docs)
    embedding_model = Zhipuembedding()
    database.get_topic_vector(embedding_model)
    database.get_entire_vector(embedding_model)
    database.persist(database_path)
    # database = Vectordatabase()
    # database.load_vector(database_path)
