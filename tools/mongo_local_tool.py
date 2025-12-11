# tools/mongo_local_tool.py

"""
MongoLocalTool -> HybridLiteratureTool
本地向量数据库 + PubMed 在线混合检索工具 (支持批量精准归位版)

功能：
1. 混合检索：同时从本地 MongoDB (Vector) 和 PubMed Online 获取证据
2. 多路召回：Gene 模式下自动生成多维查询 (机制/预后/治疗)
3. 批量处理：支持传入 genes 列表，自动循环并标记归属，实现精准溯源
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
        self.pubmed = PubMedTool()

    def _connect(self):
        """连接 MongoDB"""
        if self.client: return
        try:
            self.client = MongoClient(host=self.host, port=self.port, serverSelectionTimeoutMS=2000)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            # self.client.admin.command('ping') # 可选：检查连接
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
        # 针对肝癌 (Hepatocellular Carcinoma) 的特定查询模板
        # 也可以从 context 里传 disease 进来动态拼接
        queries = [
            ("clinical", f"{gene_name} hepatocellular carcinoma prognosis survival"),
            ("mechanism", f"{gene_name} signaling pathway liver cancer mechanism"),
            ("therapy", f"{gene_name} inhibitor therapeutic target HCC")
        ]
        
        all_results = []
        seen_hashes = set()
        
        # 每个方面只取最精华的 (本地2 + 在线1)，避免结果爆炸
        for aspect, query_text in queries:
            results = self._hybrid_search(query_text, top_k_local=2, top_k_online=1)
            
            for item in results:
                content_hash = hash(item['content'][:100])
                if content_hash not in seen_hashes:
                    item['aspect'] = aspect
                    item['matched_query'] = query_text
                    # 【关键】不要在这里加 related_gene，而在外层加，防止复用逻辑混乱
                    all_results.append(item)
                    seen_hashes.add(content_hash)
        
        return all_results

    def _generate_summary(self, results: List[Dict], subject: str, mode: str) -> str:
        """生成 Markdown 综述 (支持多基因分组)"""
        if not results:
            return f"未找到关于 {subject} 的文献证据 (本地+在线)。"
            
        lines = []
        lines.append(f"### 📚 文献检索综述: {subject}")
        lines.append(f"> 总计条目: {len(results)} \n")
        
        # 如果是批量模式，按 related_gene 分组展示
        if mode == "batch_gene":
            # 先按 gene 排序，再 groupby
            results.sort(key=lambda x: x.get('related_gene', 'Unknown'))
            for gene, gene_items in groupby(results, key=lambda x: x.get('related_gene', 'Unknown')):
                lines.append(f"#### 🧬 基因: {gene}")
                gene_items_list = list(gene_items)
                # 内部再按 aspect 分组
                gene_items_list.sort(key=lambda x: x.get('aspect', 'general'))
                for aspect, group in groupby(gene_items_list, key=lambda x: x.get('aspect', 'general')):
                    aspect_icon = {"clinical": "🏥", "mechanism": "🔬", "therapy": "💊", "general": "🔍"}.get(aspect, "📄")
                    lines.append(f"**{aspect_icon} {aspect.capitalize()}**")
                    for item in group:
                        content = item['content'].replace('\n', ' ')[:200] + "..."
                        src = item.get('source_type', 'Unknown')
                        title = item['source_metadata']['paper_title']
                        lines.append(f"- [{src}] {content} *({title})*")
                lines.append("")
        else:
            # 单基因或 Query 模式
            results.sort(key=lambda x: x.get('aspect', 'general'))
            for aspect, group in groupby(results, key=lambda x: x.get('aspect', 'general')):
                lines.append(f"**{aspect.capitalize()}**")
                for item in group:
                    content = item['content'].replace('\n', ' ')[:250] + "..."
                    lines.append(f"- {content}")
            lines.append("")
        
        return "\n".join(lines)

    def run(self, context: Dict) -> Dict:
        """
        工具入口 - 支持批量 genes 处理
        """
        print(f"[MongoLocalTool]: 正在检索文献...")
        
        gene = context.get("gene")
        genes = context.get("genes") # 获取列表参数
        query = context.get("query")
        
        results = []
        search_subject = ""
        search_mode = ""

        try:
            # === 优先处理批量基因列表 ===
            if genes and isinstance(genes, list) and len(genes) > 0:
                search_mode = "batch_gene"
                search_subject = f"Batch of {len(genes)} genes"
                # 限制批量处理数量，防止超时 (例如只查前 10 个，或全部)
                # target_genes = genes[:10] 
                target_genes = genes # 全量查询，Planner会控制传入数量
                
                print(f"  > 批量检索模式: {len(target_genes)} 个基因")
                
                for g in target_genes:
                    if not g: continue
                    # 检索单基因
                    g_res = self._search_evidence_by_gene(g)
                    
                    # 【核心修改】精准标记：为每条结果打上 related_gene 标签
                    for item in g_res:
                        item['related_gene'] = g
                    
                    results.extend(g_res)
                    print(f"    - {g}: 找到 {len(g_res)} 条证据")

            # === 处理单个基因 ===
            elif gene:
                search_subject = gene
                search_mode = "single_gene"
                logger.info(f"Running Gene Mode for: {gene}")
                results = self._search_evidence_by_gene(gene)
                for r in results: r['related_gene'] = gene # 保持一致性
                
            # === 处理通用文本查询 ===
            elif query:
                search_subject = query
                search_mode = "general_query"
                logger.info(f"Running General Mode for: {query}")
                results = self._hybrid_search(query, top_k_local=3, top_k_online=3)
                for r in results: r['aspect'] = 'general'
            
            else:
                return {"type": "search_literature", "error": "No gene/genes/query provided"}

            # 生成综述 (供 LLM 阅读)
            summary = self._generate_summary(results, search_subject, search_mode)

            # 返回结构 (raw_results 供 Planner 精准提取)
            return {
                "type": "search_literature",
                "subject": search_subject,
                "search_mode": search_mode,
                "n_results": len(results),
                "summary": summary,
                "raw_results": results, # 这里的每个 item 都必须包含 'related_gene'
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
    print("🚀 Testing Hybrid Tool (Batch Mode)...")
    
    tool = MongoLocalTool(db_name="bio", collection_name="evidence_chunks")
    # 模拟批量查询
    res = tool.run({"genes": ["TP53", "MAGEA1", "UNKNOWN_GENE_123"]})
    print(f"\n{res['summary']}")
    
    # 验证 raw_results 结构
    print("\n[Check Raw Results]:")
    for r in res['raw_results'][:3]:
        print(f"Gene: {r.get('related_gene')} | Title: {r['source_metadata']['paper_title']}")