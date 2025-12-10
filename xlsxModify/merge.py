import pandas as pd
import io


merged_df = pd.read_csv('D:/Bit/bishe/合并后的数据.csv')
df_rna = pd.read_csv('D:/Bit/bishe/xlsxModify/res_data/Count_Zhuanzhi.csv')

if 'Sample_id' in merged_df.columns:
    merged_df['Sample_id'] = merged_df['Sample_id'].astype(str).str.strip()

df_rna['Sample_id'] = df_rna['Sample_id'].astype(str).str.strip()

rna_columns = [col for col in df_rna.columns if col != 'Sample_id']
rename_dict = {col: f"RNA_{col}" for col in rna_columns}
df_rna_renamed = df_rna.rename(columns=rename_dict)
print("RNA表重命名后表头预览:")
print(df_rna_renamed.columns.tolist()[:5]) # 只看前5个

final_df = pd.merge(
    merged_df,         # 上一步合并好的表 (左边)
    df_rna_renamed,    # 处理好的 RNA 表 (右边)
    on='Sample_id',    # 这次用 Sample_id 关联
    how='inner'
)

print(f"\n最终合并表格大小: {final_df.shape}")
print("最终表格包含的列 (部分):")
cols_to_show = ['Sample_id', 'GENE_MIB2', 'RNA_TSPAN6']
print(final_df[[c for c in cols_to_show if c in final_df.columns]].head())
final_df.to_csv("最终三表合一数据.csv", index=False, encoding='utf-8-sig')






# 转置 count 表
# input_file = 'D:/Bit/bishe/xlsxModify/data/Updated_Count.xlsx'
# df = pd.read_excel(input_file, index_col=0)
# df_transposed = df.T
# output_file = 'D:/Bit/bishe/xlsxModify/res_data/Count_Zhuanzhi.csv'
# df_transposed.to_csv(output_file, encoding='utf-8-sig', index_label='Sample_id')

# print(f"成功！文件已保存为: {output_file}")