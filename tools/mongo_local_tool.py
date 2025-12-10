"""MongoLocalTool - 从本地 MongoDB 查询文献

1. 按基因关键词逐个在 MongoDB 中查询
2. 相似度排序（如果有 before_text 上下文）
3. 返回与原 PubMedTool 兼容的结构

数据约定：
- MongoDB 连接：localhost:27017, db=bio, collection=pubmed
- 文档字段：_id, title, abstract
"""

import logging
import re
from typing import List, Dict, Optional

from pymongo import MongoClient

logger = logging.getLogger(__name__)


def get_similarities(before_text: str, papers: List[Dict], top_k: int = 15) -> List[Dict]:
    """
    根据 before_text 与 papers 的相似度排序，返回 top_k 篇论文
    这里使用简单的 TF-IDF 
    Args:
        before_text: 上下文文本（如前面的查询结果或任务描述）
        papers: 论文列表 [{"title": ..., "abstract": ...}, ...]
        top_k: 返回前 k 篇
    
    Returns:
        排序后的论文列表
    """
    if not before_text or not papers:
        return papers[:top_k]
    
    # 简单相似度：计算论文 abstract 与 before_text 的共同词数
    before_words = set(before_text.lower().split())
    scored_papers = []
    for paper in papers:
        abstract = paper.get("abstract", "").lower()
        abstract_words = set(abstract.split())
        # 共同词数作为相似度
        score = len(before_words & abstract_words)
        scored_papers.append((score, paper))
    scored_papers.sort(key=lambda x: x[0], reverse=True)
    
    return [p for _, p in scored_papers[:top_k]]


class MongoLocalTool:
    
    def __init__(self, host: str = "localhost", port: int = 27017, 
                 db_name: str = "bio", collection_name: str = "pubmed"):
        """
        Args:
            host: MongoDB 主机
            port: MongoDB 端口
            db_name: 数据库名
            collection_name: 集合名
        """
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
    
    def _connect(self):
        try:
            self.client = MongoClient(host=self.host, port=self.port, serverSelectionTimeoutMS=2000)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.db.command("ping")
            logger.debug(f"Connected to MongoDB {self.host}:{self.port}/{self.db_name}/{self.collection_name}")
        except Exception as e:
            logger.exception(f"Failed to connect to MongoDB: {e}")
            raise
    
    def run(self, context: Dict) -> Dict:
        """
        Args:
            context:
                - query: 查询字符串（多个基因逗号或空格分隔，如 "BRCA1,TP53" 或 "BRCA1 TP53"）
                - before_text (可选): 用于相似度排序的上下文文本 
        Returns:
            {
                "type": "search_pubmed_mongo",
                "query": query_string,
                "results": [{"pmid": ..., "title": ..., "abstract": ...}, ...],
                "n_results": int,
                "error": str (if any)
            }
        """
        query = context.get("query")
        before_text = context.get("before_text", "")
        if not query:
            return {
                "type": "search_pubmed_mongo",
                "results": [],
                "error": "no query"
            }
        try:
            if self.collection is None:
                self._connect()
        except Exception as e:
            return {
                "type": "search_pubmed_mongo",
                "query": query,
                "results": [],
                "error": f"mongo connection error: {str(e)}"
            }

        keys = []
        for sep in [",", ";"]:
            if sep in query:
                keys = [k.strip() for k in query.split(sep) if k.strip()]
                break
        if not keys:
            keys = query.split()
        logger.debug(f"Parsed query keys: {keys}")
        
        # 逐个基因查询
        all_papers = []
        try:
            for key in keys:
                escaped_key = re.escape(key)
                mongo_query = {"abstract": {"$regex": escaped_key, "$options": "i"}}
                count = self.collection.count_documents(mongo_query)
                if count == 0:
                    logger.debug(f"No documents found for key: {key}")
                    continue
                logger.debug(f"Found {count} documents for key: {key}")
                
                # 查询结果，仅取 _id, title, abstract
                results = self.collection.find(mongo_query, {"_id": 0, "pmid": 1, "title": 1, "abstract": 1})
                papers = []
                for res in results:
                    papers.append({
                        "pmid": str(res.get("pmid", res.get("_id", ""))),
                        "title": str(res.get("title", "")),
                        "abstract": str(res.get("abstract", ""))
                    })
                all_papers.extend(papers)
        
        except Exception as e:
            logger.exception(f"Error querying MongoDB: {e}")
            return {
                "type": "search_pubmed_mongo",
                "query": query,
                "results": [],
                "error": f"mongo query error: {str(e)}"
            }
        
        # 去重（按 pmid）
        unique_papers = {}
        for paper in all_papers:
            pmid = paper["pmid"]
            if pmid not in unique_papers:
                unique_papers[pmid] = paper
        all_papers = list(unique_papers.values())
        
        # 相似度排序（如果提供了 before_text）
        if before_text:
            all_papers = get_similarities(before_text, all_papers, top_k=len(all_papers))
        
        return {
            "type": "search_pubmed_mongo",
            "query": str(query),  # 确保 query 是字符串
            "results": all_papers,
            "n_results": len(all_papers)
        }
