from databases import Vectordatabase

if __name__ == '__main__':
    from embedding import Zhipuembedding

    database = Vectordatabase()
    database.load_vector("/home/wangb/cyo/graduation/rag/databases/xihu_1")
    embedding_model = Zhipuembedding()
    query = "请帮我推荐一条西湖一日游的旅游路线，要求是：6元游船，西湖美景，便捷。除此，我还希望去曲院风荷，花港观鱼游玩。我希望我的交通时间在21分钟左右，交通距离在10千米左右。"
    route = database.query_both(query, None, embedding_model, k=4, n=1)
    for r in route:
        print('*' * 40)
        print(r)
