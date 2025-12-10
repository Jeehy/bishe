"""MongoLocalTool - 本地文献向量检索工具 (增强版)

功能：
1. 混合检索：基于 FAISS 的向量检索 + 关键词匹配
2. 多路召回：针对 Gene 自动生成多维度查询 (Prognosis/Mechanism/Drug)
3. 证据合成：自动生成 Markdown 格式的证据综述

数据约定：
- MongoDB: localhost:27017, db=bio, collection=evidence_chunks
- 依赖: sentence-transformers, faiss-cpu, numpy
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

logger = logging.getLogger(__name__)

# === 全局资源 (单例模式) ===
_GLOBAL_MODEL = None
_GLOBAL_INDEX = None
_GLOBAL_DOC_MAP = []

class MongoLocalTool:
    
    def __init__(self, host: str = "localhost", port: int = 27017, 
                 db_name: str = "bio", collection_name: str = "evidence_chunks"):
        """
        初始化配置
        """
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None

    def _connect(self):
        """连接数据库"""
        if self.client: return
        try:
            self.client = MongoClient(host=self.host, port=self.port, serverSelectionTimeoutMS=2000)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
        except Exception as e:
            logger.exception(f"Failed to connect to MongoDB: {e}")
            raise

    def _ensure_resources(self):
        """
        加载模型与构建索引。
        耗时操作，仅在首次调用时执行。
        """
        global _GLOBAL_MODEL, _GLOBAL_INDEX, _GLOBAL_DOC_MAP
        self._connect()
        
        # 1. 加载模型
        if _GLOBAL_MODEL is None:
            logger.info(">>> [MongoLocalTool] Loading model (all-MiniLM-L6-v2)...")
            try:
                _GLOBAL_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                raise

        # 2. 构建索引 (从 MongoDB 读取向量)
        if _GLOBAL_INDEX is None:
            logger.info(">>> [MongoLocalTool] Building FAISS index from MongoDB...")
            start_time = time.time()
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
                        'source': doc.get('source_filename', '')
                    })
            
            if vectors:
                vector_matrix = np.array(vectors)
                dimension = vector_matrix.shape[1]
                # 使用内积 (Inner Product) 索引，假设向量已归一化则等同于余弦相似度
                index = faiss.IndexFlatIP(dimension)
                index.add(vector_matrix)
                _GLOBAL_INDEX = index
                _GLOBAL_DOC_MAP = doc_map
                logger.info(f">>> [MongoLocalTool] Index built with {len(vectors)} chunks in {time.time()-start_time:.2f}s")
            else:
                logger.warning(">>> [MongoLocalTool] No vectors found in database! Search will return empty.")
                _GLOBAL_INDEX = None
                _GLOBAL_DOC_MAP = []

    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """关键词覆盖率打分"""
        if not query or not text: return 0.0
        q_terms = set(re.findall(r'\w+', query.lower()))
        t_terms = set(re.findall(r'\w+', text.lower()))
        if not q_terms: return 0.0
        return len(q_terms.intersection(t_terms)) / len(q_terms)

    def _search_core(self, query: str, top_k: int = 8, alpha: float = 0.7, fetch_k: int = 50) -> List[Dict]:
        """
        核心检索方法：FAISS Vector + 关键词混合打分
        """
        self._ensure_resources()
        if _GLOBAL_INDEX is None or not _GLOBAL_DOC_MAP:
            return []

        # 1. 向量检索 (Vector Search)
        query_vector = _GLOBAL_MODEL.encode([query])
        query_vector = np.array(query_vector, dtype='float32')
        # 检索 top fetch_k 个向量候选
        D, I = _GLOBAL_INDEX.search(query_vector, min(fetch_k, len(_GLOBAL_DOC_MAP)))
        
        candidates = []
        for rank, idx in enumerate(I[0]):
            if idx == -1: continue
            doc_data = _GLOBAL_DOC_MAP[idx]
            # 向量分数
            vec_score = float(D[0][rank])
            # 关键词分数
            kw_score = self._calculate_keyword_score(query, doc_data['text'])
            # 2. 混合打分 (Hybrid Scoring)
            # alpha 控制向量检索权重的占比
            hybrid_score = (alpha * vec_score) + ((1 - alpha) * kw_score)
            
            # 3. 章节加权 (Section Boosting)
            # 优先展示结果与讨论部分
            section = str(doc_data['section']).lower()
            multiplier = 1.0
            if any(x in section for x in ['result', 'discussion', 'conclusion']):
                multiplier = 1.2
            elif 'abstract' in section:
                multiplier = 1.1
            final_score = hybrid_score * multiplier
            candidates.append({
                "content": doc_data['text'],
                "source_metadata": {
                    "paper_title": doc_data['paper_title'],
                    "section": doc_data['section'],
                    "filename": doc_data['source']
                },
                "scores": {
                    "final": round(final_score, 4),
                    "vector": round(vec_score, 4),
                    "keyword": round(kw_score, 4)
                }
            })

        # 4. 重新排序
        candidates.sort(key=lambda x: x['scores']['final'], reverse=True)
        return candidates[:top_k]

    def _search_evidence_by_gene(self, gene_name: str) -> List[Dict]:
        """
        针对特定基因的多路召回策略
        生成 3 个不同侧重点的 Query，分别调用检索
        """
        queries = [
            ("clinical_prognosis", f"{gene_name} high expression prognosis survival rate HCC"),
            ("mechanism", f"{gene_name} signaling pathway mechanism proliferation invasion liver cancer"),
            ("drug_therapy", f"{gene_name} inhibitor therapeutic target drug efficacy HCC")
        ]
        
        all_results = []
        seen_contents = set()
        
        for aspect, query_text in queries:
            # 分别检索，每个维度取 Top 3
            results = self._search_core(query_text, top_k=5, alpha=0.7, fetch_k=30)
            
            for item in results:
                # 简单去重 (取前50字符哈希)
                signature = item['content'][:50]
                if signature not in seen_contents:
                    item['aspect'] = aspect
                    item['matched_query'] = query_text
                    all_results.append(item)
                    seen_contents.add(signature)
        
        return all_results

    def _generate_summary(self, results: List[Dict], subject: str) -> str:
        """
        生成 Markdown 格式的证据综述
        """
        if not results:
            return f"未找到关于 {subject} 的文献证据。"
            
        # 分组 (general / clinical_prognosis / mechanism / drug_therapy)
        results.sort(key=lambda x: x.get('aspect', 'general'))
        lines = []
        lines.append(f"### 📚 {subject} 文献证据综述")
        lines.append(f"> 共检索到 {len(results)} 条相关证据片段。\n")
        
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
                # 截断过长文本
                if len(content) > 300:
                    content = content[:300] + "..."
                source = item['source_metadata']['paper_title']
                score = item['scores']['final']
                lines.append(f"- {content} *[Score: {score:.2f} | Src: {source}]*")
            lines.append("") 
        
        return "\n".join(lines)

    def run(self, context: Dict) -> Dict:
        """
        工具统一入口
        """
        # 1. 参数解析
        gene = context.get("gene")
        if not gene and context.get("genes"):
            gene = context.get("genes")[0]
        query = context.get("query")
        results = []
        search_subject = ""
        search_mode = ""

        try:
            if gene:
                # === 路径 A: 基因多路召回模式 ===
                search_subject = gene
                search_mode = "gene_evidence_mining"
                logger.info(f"Running Gene Mode for: {gene}")
                results = self._search_evidence_by_gene(gene)
                
            elif query:
                # === 路径 B: 通用查询混合检索模式 ===
                search_subject = query
                search_mode = "general_hybrid_search"
                logger.info(f"Running Query Mode for: {query}")
                results = self._search_core(query, top_k=5)
                # 标记 aspect 以便生成 summary
                for r in results:
                    r['aspect'] = 'general'
            
            else:
                return {
                    "type": "search_literature",
                    "error": "No 'gene' or 'query' provided."
                }

            # 3. 生成 Markdown 综述
            summary = self._generate_summary(results, search_subject)

            # 4. 返回结果
            return {
                "type": "search_literature",
                "subject": search_subject,
                "search_mode": search_mode,
                "n_results": len(results),
                "summary": summary,       # <--- LLM 核心阅读内容
                "raw_results": results,   # <--- 保留原始数据结构
                "error": None
            }

        except Exception as e:
            logger.exception(f"Error in MongoLocalTool run: {e}")
            return {
                "type": "search_literature",
                "error": str(e),
                "summary": f"检索出错: {str(e)}",
                "results": []
            }


# =============================================================================
# 验证用的 Main 函数 (Self-Check)
# =============================================================================
if __name__ == "__main__":
    import sys
    
    # 配置控制台日志，方便观察内部流程
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    print("\n🚀 [MongoLocalTool] 启动自检程序 (Vector Ready Mode)...\n")
    
    # 1. 初始化工具
    # 确保 MongoDB 服务已开启，且 bio.evidence_chunks 集合中有带 vector 字段的数据
    try:
        print("Creating tool instance...")
        tool = MongoLocalTool(db_name="bio", collection_name="evidence_chunks")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 测试场景 1: Gene 模式 (核心功能)
    # 预期: 自动生成多条查询，执行多次检索，最后合并生成综述
    # ---------------------------------------------------------
    test_gene = "TP53"
    print(f"\n" + "="*60)
    print(f"🧪 [Test 1] Testing Gene Mode for '{test_gene}'")
    print("预期行为: 触发 gene_evidence_mining 模式，进行多路召回")
    print("="*60)
    
    start_time = time.time()
    res1 = tool.run({"gene": test_gene})
    duration = time.time() - start_time
    
    if res1.get("error"):
        print(f"❌ Error: {res1['error']}")
    else:
        print(f"✅ Success! (耗时 {duration:.2f}s)")
        print(f"   - Search Mode: {res1.get('search_mode')} (应为 gene_evidence_mining)")
        print(f"   - Total Results: {res1.get('n_results')}")
        
        # 检查是否真的用到了向量检索 (检查第一条结果是否有 scores.vector)
        if res1.get('raw_results') and 'vector' in res1['raw_results'][0]['scores']:
            vec_score = res1['raw_results'][0]['scores']['vector']
            print(f"   - Vector Score Example: {vec_score:.4f} (证明使用了向量检索)")
        else:
            print("   ⚠️ Warning: 未检测到向量分数，请检查数据库 vector 字段")

        print("\n📝 [Summary Preview]:")
        print("-" * 40)
        # 打印摘要的前 500 个字符
        print(res1.get('summary', '')[:500].replace('\n', ' ') + "...") 
        print("-" * 40)

    # ---------------------------------------------------------
    # 测试场景 2: Query 模式 (通用检索)
    # 预期: 对输入语句进行单次混合检索
    # ---------------------------------------------------------
    test_query = "liver cancer immunotherapy efficacy"
    print(f"\n" + "="*60)
    print(f"🧪 [Test 2] Testing Query Mode for '{test_query}'")
    print("预期行为: 触发 general_hybrid_search 模式")
    print("="*60)
    
    res2 = tool.run({"query": test_query})
    
    if res2.get("error"):
        print(f"❌ Error: {res2['error']}")
    else:
        print(f"✅ Success!")
        print(f"   - Search Mode: {res2.get('search_mode')} (应为 general_hybrid_search)")
        print(f"   - Total Results: {res2.get('n_results')}")
        
        print("\n📝 [Summary Preview]:")
        print("-" * 40)
        print(res2.get('summary', '')[:300].replace('\n', ' ') + "...")
        print("-" * 40)

    print("\n🏁 自检完成。")