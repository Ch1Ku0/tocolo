import json
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from collections import defaultdict
import math

def get_best_matching_keys(query, corpus, method_path_count, top_n=10):
    # 将每个文档分词并存储在corpus_tokens中
    corpus_tokens = {key: word_tokenize(value.lower()) for key, value in corpus.items()}

    # 创建BM25对象
    bm25 = BM25Okapi(list(corpus_tokens.values()))

    # 查询的分词
    query_tokens = word_tokenize(query.lower())

    # 计算每个文档的BM25分数
    scores = bm25.get_scores(query_tokens)


    weighted_scores = []

    # 对数
    # for idx, score in enumerate(scores):
    #     doc_key = list(corpus.keys())[idx]
    #     weight = method_path_count.get(doc_key, 1)
    #     smoothed_weight = 1 + math.log(1 + weight)  # 平滑处理
    #     weighted_scores.append(score * smoothed_weight)

    # 平方根
    for idx, score in enumerate(scores):
        doc_key = list(corpus.keys())[idx]
        weight = method_path_count.get(doc_key, 1)
        smoothed_weight = 1 + math.sqrt(weight)  # 平滑处理
        weighted_scores.append(score * smoothed_weight)
    
    # Min-Max 归一化
    # min_weight = min(method_path_count.values())
    # max_weight = max(method_path_count.values())
    # for idx, score in enumerate(scores):
    #     doc_key = list(corpus.keys())[idx]
    #     weight = method_path_count.get(doc_key, 1)
    #     smoothed_weight = 1 + (weight - min_weight) / (max_weight - min_weight)  # 归一化到 [1,2]
    #     weighted_scores.append(score * smoothed_weight)

    # 获取按加权分数排序后的文档键
    sorted_keys = sorted(corpus.keys(), key=lambda k: weighted_scores[list(corpus.keys()).index(k)], reverse=True)

    # 返回前top_n个匹配的文档
    return sorted_keys[:top_n]
