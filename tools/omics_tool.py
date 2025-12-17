# tools/omics_tool.py
import pandas as pd
import os

class OmicsTool:
    """
    基于真实 DESeq2 差异表达文件的 Omics 工具
    
    模式 A (Discovery): 不传 genes 参数 -> 返回 Top N 显著差异基因
    模式 B (Verification): 传入 genes 参数 -> 查询特定基因的表达数值和显著性
    """

    def __init__(self, de_path="D:/Bit/bishe/data/DESeq2_results_all.csv"):
        self.de_path = de_path

    def run(self, context):
        # 1. 基础检查
        if not os.path.exists(self.de_path):
            return {"type":"run_omics","error":"DESeq2 文件不存在"}

        try:
            df = pd.read_csv(self.de_path)
        except Exception as e:
            return {"type":"run_omics","error":f"读取CSV失败: {str(e)}"}

        required = {"gene_id","log2FoldChange","padj"}
        if not required.issubset(df.columns):
            return {"type":"run_omics","error":"差异表达文件缺少必要列(gene_id, log2FoldChange, padj)"}

        # 获取上下文中的特定基因列表
        target_genes = context.get("genes", [])

        # ==========================================
        # 模式 B: 验证模式 (Verification Mode)
        # 场景：中间步骤，验证特定基因
        # ==========================================
        if target_genes:
            print(f"    🧪 [OmicsTool] 进入验证模式，查询基因: {target_genes}")
            verification_results = []
            
            for gene in target_genes:
                # 模糊匹配或精确匹配，这里用精确匹配，注意大小写通常需要一致
                # 如果担心大小写问题，可以将两边都 .str.upper()
                match = df[df["gene_id"] == gene]
                
                if match.empty:
                    verification_results.append({
                        "gene": gene,
                        "found": False,
                        "note": "未在测序结果中找到该基因"
                    })
                else:
                    row = match.iloc[0]
                    log2fc = float(row["log2FoldChange"])
                    padj = float(row["padj"])
                    is_sig = padj < 0.05 and abs(log2fc) > 1.0 # 定义显著性阈值
                    
                    verification_results.append({
                        "gene": gene,
                        "found": True,
                        "log2FoldChange": round(log2fc, 4),
                        "padj": padj, # 科学计数法通常由 JSON 序列化处理
                        "is_significant": is_sig,
                        "regulation": "Up" if log2fc > 0 else "Down"
                    })

            return {
                "type": "run_omics_verification",
                "results": verification_results,
                "description": f"已查询 {len(target_genes)} 个基因的表达情况"
            }

        # ==========================================
        # 模式 A: 发现模式 (Discovery Mode)
        # 场景：第一步，寻找线索
        # ==========================================
        else:
            print(f"    🔭 [OmicsTool] 进入发现模式，寻找 Top 显著基因")
            # 过滤显著的
            sig = df[df["padj"] < 0.05]
            
            # 分别取 Top 上调和下调
            up = sig.sort_values("log2FoldChange", ascending=False).head(50)
            down = sig.sort_values("log2FoldChange", ascending=True).head(50)

            return {
                "type": "run_omics_discovery",
                "results": {
                    "top_upregulated": up[["gene_id","log2FoldChange", "padj"]].to_dict("records"),
                    "top_downregulated": down[["gene_id","log2FoldChange", "padj"]].to_dict("records")
                },
                "n_significant_total": len(sig),
                "description": "已返回 Top 50 上调和 Top 50 下调基因"
            }

# --- 运行验证 ---
if __name__ == "__main__":
    tool = OmicsTool()
    
    # 测试场景 1: 发现模式
    print("--- Discovery Mode ---")
    print(tool.run({}))
    
    # 测试场景 2: 验证模式
    print("\n--- Verification Mode ---")
    # 假设查询一个存在的基因 (你需要换成你 CSV 里真实的基因名) 和一个不存在的
    print(tool.run({"genes": ["TP53"]}))