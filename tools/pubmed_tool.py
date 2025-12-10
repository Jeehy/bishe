"""tools/pubmed_tool.py - 在线 PubMed 检索工具

封装 Biopython 的 Entrez API 调用，用于实时检索 PubMed 文献。
"""

import logging
import re
from typing import List, Dict
from Bio import Entrez
Entrez.email = "826329938@qq.com"


logger = logging.getLogger(__name__)

class PubMedTool:
    def __init__(self, email: str = None):
        """
        初始化 PubMed 工具
        :param email: 可选，覆盖默认的 Entrez.email
        """
        if email:
            self.email = email
            Entrez.email = email

    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        使用 Biopython 查询 PubMed
        Args:
            query: 查询字符串
            max_results: 最大返回结果数
        Returns:
            List[Dict]: 结构化文献列表
        """
        
        logger.info(f"🔍 [PubMedTool] Searching Online for: {query}")
        results = []
        try:
            # Step 1: ESearch 获取 ID
            # sort="relevance" 确保返回最相关的文献
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]
            
            if not id_list:
                return []

            # Step 2: EFetch 获取详细信息 (MEDLINE 格式易于解析)
            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
            records = handle.read().split("\n\n")
            handle.close()

            for rec in records:
                if not rec.strip(): continue
                
                # 简单正则解析 Title (TI) 和 Abstract (AB)
                title_match = re.search(r"TI\s+-\s+(.*?)\n[A-Z]", rec, re.DOTALL)
                abs_match = re.search(r"AB\s+-\s+(.*?)\n[A-Z]", rec, re.DOTALL)
                
                # 清洗换行符
                title = title_match.group(1).replace("\n      ", " ") if title_match else "Unknown Title"
                abstract = abs_match.group(1).replace("\n      ", " ") if abs_match else ""
                
                # 只有当存在摘要时才保留
                if abstract:
                    results.append({
                        "content": abstract,
                        "source_metadata": {
                            "paper_title": title,
                            "section": "Abstract",
                            "filename": "PubMed Online"
                        },
                        # 给予在线结果一个固定的高分，确保它们在后续混合排序中有一席之地
                        "scores": {"final": 0.95}, 
                        "source_type": "Online"
                    })
        except Exception as e:
            logger.error(f"PubMed online search failed: {e}")
        
        return results

# === 独立测试入口 ===
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    print("🚀 Testing PubMedTool independently...")
    
    tool = PubMedTool()
    res = tool.search("liver cancer immunotherapy novel target", max_results=2)
    
    for i, r in enumerate(res):
        print(f"\n[{i+1}] {r['source_metadata']['paper_title']}")
        print(f"    {r['content'][:150]}...")