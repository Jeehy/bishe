"""MongoLocalTool -> HybridLiteratureTool
本地向量数据库 + PubMed 在线混合检索工具 (重构版)

功能：
1. 混合检索：同时从本地 MongoDB (Vector) 和 PubMed Online 获取证据
2. 多路召回：Gene 模式下自动生成多维查询
3. 鲁棒性：本地或在线任一渠道失败不影响整体运行

依赖: 
- sentence-transformers, faiss-cpu, numpy, pymongo
- PubMedTool (tools.pubmed_tool) <--- 新增依赖
"""

import logging
import re
import time
import numpy as np
import faiss
from typing import List, Dict
from itertools import groupby
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tools.pubmed_tool import PubMedTool

logger = logging.getLogger(__name__)

# === 全局资源 (单例模式) ===
_GLOBAL_MODEL = None
_GLOBAL_INDEX = None
_GLOBAL_DOC_MAP = []

class MongoLocalTool:
    
    def __init__(self, host: str = "localhost", port: int = 27017, 
                 db_name: str = "bio", collection_name: str = "evidence_chunks"):
        """
        初始化工具
        """
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        
        # 初始化 PubMed 在线工具
        # 建议在此处或 executor 中统一管理邮箱配置
        self.pubmed = PubMedTool(email="your_email@example.com")

    def _connect(self):
        """连接 MongoDB"""
        if self.client: return
        try:
            self.client = MongoClient(host=self.host, port=self.port, serverSelectionTimeoutMS=2000)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.client.admin.command('ping')
            logger.debug(f"Connected to MongoDB {self.host}:{self.port}/{self.db_name}")
        except Exception as e:
            logger.exception(f"Failed to connect to MongoDB: {e}")

    def _ensure_resources(self):
        """加载模型与构建索引"""
        global _GLOBAL_MODEL, _GLOBAL_INDEX, _GLOBAL_DOC_MAP
        
        self._connect()
        
        if _GLOBAL_MODEL is None:
            try:
                logger.info(">>> [HybridTool] Loading model (all-MiniLM-L6-v2)...")
                _GLOBAL_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")

        if _GLOBAL_INDEX is None and self.collection is not None:
            try:
                logger.info(">>> [HybridTool] Building FAISS index from MongoDB...")
                start_time = time.time()
                # 只读取带向量的数据
                cursor = self.collection.find(
                    {"vector": {"$exists": True}},
                    {"vector": 1, "text": 1, "section": 1, "paper_title": 1, "source_filename": 1}
                )
                
                vectors = []
                doc_map = []
                for doc in cursor:
                    vec = doc.get('vector')
                    if vec and len(vec) > 0:
                        vectors.append(np.array(vec, dtype='float32'))
                        doc_map.append({
                            'id': str(doc.get('_id')),
                            'text': doc.get('text', ''),
                            'section': doc.get('section', 'Unknown'),
                            'paper_title': doc.get('paper_title', 'Unknown'),
                            'source': doc.get('source_filename', 'LocalDB')
                        })
                
                if vectors:
                    vector_matrix = np.array(vectors)
                    dimension = vector_matrix.shape[1]
                    index = faiss.IndexFlatIP(dimension)
                    index.add(vector_matrix)
                    _GLOBAL_INDEX = index
                    _GLOBAL_DOC_MAP = doc_map
                    logger.info(f">>> Index built with {len(vectors)} chunks in {time.time()-start_time:.2f}s")
                else:
                    logger.warning(">>> No vectors found in local database.")
                    _GLOBAL_INDEX = None
                    _GLOBAL_DOC_MAP = []
            except Exception as e:
                logger.warning(f"Failed to build local index: {e}")

    def _calculate_keyword_score(self, query: str, text: str) -> float:
        if not query or not text: return 0.0
        q_terms = set(re.findall(r'\w+', query.lower()))
        t_terms = set(re.findall(r'\w+', text.lower()))
        if not q_terms: return 0.0
        return len(q_terms.intersection(t_terms)) / len(q_terms)

    # === 1. 本地检索核心 ===
    def _search_local_core(self, query: str, top_k: int = 5) -> List[Dict]:
        self._ensure_resources()
        if _GLOBAL_INDEX is None or not _GLOBAL_DOC_MAP:
            return []

        try:
            query_vector = _GLOBAL_MODEL.encode([query])
            query_vector = np.array(query_vector, dtype='float32')
            D, I = _GLOBAL_INDEX.search(query_vector, min(50, len(_GLOBAL_DOC_MAP)))
            
            candidates = []
            for rank, idx in enumerate(I[0]):
                if idx == -1: continue
                doc_data = _GLOBAL_DOC_MAP[idx]
                vec_score = float(D[0][rank])
                kw_score = self._calculate_keyword_score(query, doc_data['text'])
                
                # 混合打分
                hybrid_score = (0.7 * vec_score) + (0.3 * kw_score)
                
                # 章节加权
                section = str(doc_data['section']).lower()
                multiplier = 1.0
                if any(x in section for x in ['result', 'discussion', 'conclusion']):
                    multiplier = 1.2
                elif 'abstract' in section:
                    multiplier = 1.1
                
                candidates.append({
                    "content": doc_data['text'],
                    "source_metadata": {
                        "paper_title": doc_data['paper_title'],
                        "section": doc_data['section'],
                        "filename": doc_data['source']
                    },
                    "scores": {"final": round(hybrid_score * multiplier, 4)},
                    "source_type": "Local" # 标记来源
                })
            
            candidates.sort(key=lambda x: x['scores']['final'], reverse=True)
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return []

    # === 2. 混合执行逻辑 (Local + PubMed) ===
    def _hybrid_search(self, query: str, top_k_local: int = 5, top_k_online: int = 5) -> List[Dict]:
        """合并本地和在线结果"""
        # 1. 本地检索
        local_res = self._search_local_core(query, top_k=top_k_local)
        
        # 2. 在线检索 (调用外部工具)
        online_res = self.pubmed.search(query, max_results=top_k_online)
        
        # 3. 合并
        combined = local_res + online_res
        return combined

    def _search_evidence_by_gene(self, gene_name: str) -> List[Dict]:
        """针对 Gene 的多路混合召回"""
        queries = [
            ("clinical_prognosis", f"{gene_name} hepatocellular carcinoma prognosis survival"),
            ("mechanism", f"{gene_name} signaling pathway liver cancer mechanism"),
            ("drug_therapy", f"{gene_name} inhibitor therapeutic target HCC")
        ]
        
        all_results = []
        seen_hashes = set()
        
        for aspect, query_text in queries:
            # 混合检索：本地 3 条 + 在线 2 条
            results = self._hybrid_search(query_text, top_k_local=3, top_k_online=2)
            
            for item in results:
                # 去重
                content_hash = hash(item['content'][:100])
                if content_hash not in seen_hashes:
                    item['aspect'] = aspect
                    item['matched_query'] = query_text
                    all_results.append(item)
                    seen_hashes.add(content_hash)
        
        return all_results

    def _generate_summary(self, results: List[Dict], subject: str) -> str:
        """生成 Markdown 综述"""
        if not results:
            return f"未找到关于 {subject} 的文献证据 (本地+在线)。"
            
        results.sort(key=lambda x: x.get('aspect', 'general'))
        
        lines = []
        lines.append(f"### 📚 {subject} 文献证据综述 (混合检索)")
        local_count = sum(1 for r in results if r.get('source_type') == 'Local')
        online_count = sum(1 for r in results if r.get('source_type') == 'Online')
        lines.append(f"> 检索结果: {len(results)} 条 (本地: {local_count}, 在线 PubMed: {online_count})\n")
        
        for aspect, group in groupby(results, key=lambda x: x.get('aspect', 'general')):
            title_map = {
                "clinical_prognosis": "🏥 临床预后 (Prognosis)",
                "mechanism": "🔬 分子机制 (Mechanism)",
                "drug_therapy": "💊 药物治疗 (Therapy)",
                "general": "🔍 通用检索结果"
            }
            display_title = title_map.get(aspect, aspect.capitalize())
            lines.append(f"**{display_title}**")
            
            for item in group:
                content = item['content'].replace('\n', ' ')
                if len(content) > 300: content = content[:300] + "..."
                
                title = item['source_metadata']['paper_title']
                src_type = item.get('source_type', 'Local')
                icon = "🏠" if src_type == "Local" else "🌐"
                
                lines.append(f"- {icon} [{src_type}] {content} *[Src: {title}]*")
            lines.append("")
        
        return "\n".join(lines)

    def run(self, context: Dict) -> Dict:
        """
        工具入口
        """
        print(f"[MongoLocalTool]: 正在检索文献\n")
        gene = context.get("gene")
        if not gene and context.get("genes"): gene = context.get("genes")[0]
        query = context.get("query")
        
        results = []
        search_subject = ""
        search_mode = ""

        try:
            if gene:
                search_subject = gene
                search_mode = "gene_hybrid_mining"
                logger.info(f"Running Gene Hybrid Mode for: {gene}")
                results = self._search_evidence_by_gene(gene)
                
            elif query:
                search_subject = query
                search_mode = "general_hybrid_search"
                logger.info(f"Running General Hybrid Mode for: {query}")
                results = self._hybrid_search(query, top_k_local=3, top_k_online=3)
                for r in results: r['aspect'] = 'general'
            
            else:
                return {"type": "search_literature", "error": "No gene/query provided"}

            summary = self._generate_summary(results, search_subject)

            return {
                "type": "search_literature",
                "subject": search_subject,
                "search_mode": search_mode,
                "n_results": len(results),
                "summary": summary,
                "raw_results": results,
                "error": None
            }

        except Exception as e:
            logger.exception(f"Error in Hybrid Tool: {e}")
            return {
                "type": "search_literature",
                "error": str(e),
                "summary": f"检索出错: {str(e)}",
                "results": []
            }

# === 验证 ===
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("🚀 Testing Hybrid Tool (Local + PubMed)...")
    
    try:
        # 请确保 MongoDB 有 evidence_chunks 集合，或者它会优雅降级只显示 Online 结果
        tool = MongoLocalTool(db_name="bio", collection_name="evidence_chunks")
        res = tool.run({"gene": "TP53"})
        print(f"\n{res['summary']}")
    except Exception as e:
        print(f"Failed: {e}")