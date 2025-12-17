# tools/mongo_local_tool.py

"""
MongoLocalTool -> HybridLiteratureTool
本地向量数据库 + PubMed 在线混合检索工具 (支持批量精准归位版)

功能：
1. 混合检索：同时从本地 MongoDB (Vector) 和 PubMed Online 获取证据
2. 多路召回：Gene 模式下自动生成多维查询 (机制/预后/治疗)
3. 批量处理：支持传入 genes 列表，自动循环并标记归属，实现精准溯源
"""

import logging, re, time, faiss, json
import numpy as np
from typing import List, Dict
from itertools import groupby
from pymongo import MongoClient
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tools.pubmed_tool import PubMedTool
from tools.summary_tool import summary

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
        工具入口 - 支持批量 genes 处理 (带 LLM 自动总结和文件转储)
        """
        print(f"    [MongoLocalTool]: 正在检索文献...")
        
        gene = context.get("gene")
        genes = context.get("genes")
        query = context.get("query")
        summaries_list = [] # 🆕 用于存储精简后的总结，传给 LLM
        all_raw_evidence = [] # 🆕 用于存储所有原始证据，保存到文件
        gene_details_map = {}
        search_subject = ""
        
        try:
            # === 优先处理批量基因列表 ===
            if genes and isinstance(genes, list) and len(genes) > 0:
                search_mode = "batch_gene"
                search_subject = f"Batch of {len(genes)} genes"
                target_genes = genes 
                
                print(f"  > 批量检索模式: {len(target_genes)} 个基因")
                
                for idx, g in enumerate(target_genes):
                    if not g: continue
                    print(f"    [{idx+1}/{len(target_genes)}] 正在检索并总结: {g} ...", end="", flush=True)
                    
                    g_res = self._search_evidence_by_gene(g)
                    for item in g_res: item['related_gene'] = g
                    all_raw_evidence.extend(g_res)
                    
                    # 生成总结
                    summary_text = "未检索到相关文献"
                    if g_res:
                        evidence_text = "\n".join([f"- {r['content'][:300]}..." for r in g_res[:5]])
                        # 提示词微调：要求更简练，方便展示
                        prompt = (
                            f"根据以下基因【{g}】与肝癌文献片段，用几句话概括其作用(机制/预后/治疗)。"
                            f"不要换行，80字以内。\n片段：\n{evidence_text}"
                        )
                        try:
                            summary_text = summary(prompt).strip()
                            print(" ✅ 总结完成")
                        except Exception as e:
                            summary_text = "(总结生成失败)"
                            print(f" ⚠️ 总结失败")
                    else:
                        print(" ⚠️ 无文献")

                    summaries_list.append(f"**{g}**: {summary_text}")
                    
                    # 🆕 记录结构化详情
                    gene_details_map[g] = {
                        "count": len(g_res),
                        "summary": summary_text
                    }

            # === 处理单个基因 ===
            elif gene:
                # ... (同理处理单基因) ...
                search_subject = gene
                g_res = self._search_evidence_by_gene(gene)
                all_raw_evidence.extend(g_res)
                
                evidence_text = "\n".join([f"- {r['content'][:300]}..." for r in g_res[:5]])
                prompt = f"用一句话概括基因 {gene} 在肝癌中的作用。基于：\n{evidence_text}"
                s_text = summary(prompt).strip()
                summaries_list.append(f"**{gene}**: {s_text}")
                
                gene_details_map[gene] = {
                    "count": len(g_res),
                    "summary": s_text
                }
            
            # ... (Query模式略) ...

            # 保存原始证据到文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            evidence_filename = f"literature_raw_{timestamp}.json"
            with open(evidence_filename, "w", encoding="utf-8") as f:
                json.dump({
                    "task_genes": genes if genes else [gene],
                    "total_hits": len(all_raw_evidence),
                    "details": all_raw_evidence
                }, f, ensure_ascii=False, indent=2)
            
            final_summary_str = "\n".join(summaries_list)
            
            return {
                "type": "search_literature",
                "subject": search_subject,
                "n_results": len(all_raw_evidence), # 总数仅供参考
                "summary": final_summary_str,
                "gene_details": gene_details_map,   # 🆕 关键：返回这个 map 给 Planner
                "raw_results_file": evidence_filename,
                "error": None
            }

        except Exception as e:
            # ... (异常处理) ...
            return {"type": "search_literature", "error": str(e)}