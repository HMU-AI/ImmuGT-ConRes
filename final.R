############################################################
## 0. 环境 & 工作目录（原代码不变）
############################################################
setwd("/mnt/disk4data/result/raw_data/STAD/GSE62254")
getwd()

############################################################
## 1. 安装 & 加载依赖（【新增HGNChelper安装】，重中之重！，修复加载逻辑）
############################################################
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(
  c("affy", "annotate", "hgu133plus2.db"),
  ask = FALSE,
  update = FALSE
)

# ============ 【新增】安装新版HGNChelper,安了就不用安了 ============
if(!require("devtools")) install.packages("devtools")
devtools::install_github("waldronlab/HGNChelper", force = TRUE) # 强制安装最新版

# ============ 加载所有包 ============
library(affy)
library(hgu133plus2.db)
library(dplyr)
library(HGNChelper) # 加载基因名校正包

############################################################
## 2. 读取 CEL 文件（原代码不变）
############################################################
cel_path <- "/mnt/disk2/wym/raw_data/STAD/GSE62254_RAW"
raw_data <- ReadAffy(celfile.path = cel_path)

############################################################
## 3. RMA 标准化（背景校正 + 分位数归一化 + log2，原代码不变）
############################################################
eset <- rma(raw_data)
expr_matrix <- exprs(eset)
dim(expr_matrix)   # 54675 × 300

############################################################
## 4. 规范样本名（尽早做：只保留 GSM，原代码不变）
############################################################
colnames(expr_matrix) <- sub(
  "(GSM[0-9]+).*",
  "\\1",
  colnames(expr_matrix)
)
stopifnot(!any(duplicated(colnames(expr_matrix))))

############################################################
## 5. 探针 → 基因（GPL570，原代码不变）
############################################################
probe2gene <- AnnotationDbi::select(
  hgu133plus2.db,
  keys = rownames(expr_matrix),
  columns = "SYMBOL",
  keytype = "PROBEID"
)

expr_df <- as.data.frame(expr_matrix)
expr_df$PROBEID <- rownames(expr_df)

expr_gene <- merge(probe2gene, expr_df, by = "PROBEID")
expr_gene <- expr_gene[!is.na(expr_gene$SYMBOL), ]

############################################################
## 6. 多探针合并为基因（取均值，原代码不变）
############################################################
expr_gene_final <- expr_gene %>%
  group_by(SYMBOL) %>%
  summarise(across(where(is.numeric), mean)) %>% 
  ungroup() # 新增解除分组，避免后续报错

############################################################
## ============ 【核心新增】7. HGNChelper 基因名标准化校正（★★★最关键步骤★★★）
## 位置：探针合并后，基因列表筛选前，必加！！！修复原代码语法错误
############################################################
# 提取需要校正的基因名向量
gene_names_to_fix <- expr_gene_final$SYMBOL

# 执行基因名校正：人类基因，核心函数checkGeneSymbols
# 返回结果是data.frame，包含：原基因名、是否合规、校正后标准名
corrected_gene_res <- checkGeneSymbols(
  x = gene_names_to_fix,
  species = "human", # 人类胃癌STAD，固定human
  unmapped.as.na = FALSE # 保留无法识别的基因名（重要！避免有效基因被删）
)

# ============ 精准筛选：只看 被修改过 的基因名 ============
print("===== 所有被修改校正的基因名明细（原始名 → 标准名） =====")
modified_genes <- corrected_gene_res[corrected_gene_res$Approved == FALSE, ]
# 只保留有校正结果的（排除NA的无效基因）
modified_genes <- modified_genes[!is.na(modified_genes$Suggested.Symbol), ]
# 只看名字发生实际变更的基因
real_modified <- modified_genes[modified_genes$x != modified_genes$Suggested.Symbol, ]
print(real_modified)

# ============ 关键：将校正后的标准基因名替换回原矩阵 ============
expr_gene_final$SYMBOL <- corrected_gene_res$Suggested.Symbol

# ============ 清理：删除校正后为NA的基因（这些是无效基因名，无官方注释） ============
expr_gene_final <- expr_gene_final %>% filter(!is.na(SYMBOL))

# ============ 去重：极少数情况一个标准基因名对应多个原基因，再次按基因名取均值去重 ============
expr_gene_final <- expr_gene_final %>%
  group_by(SYMBOL) %>%
  summarise(across(where(is.numeric), mean)) %>% 
  ungroup()

############################################################
## 8. 读取 mRNA 基因列表（作为绝对模板，决定最终顺序+补0依据，核心修改）
##############################################################
gene_list <- read.csv(
  "/home/wangshaoxiang/wsx_R/human_mrna_genes_CORRECTED.csv",
  stringsAsFactors = FALSE
)
gene_order <- gene_list$external_gene_name
# 构建【基因顺序绝对模板】，保证后续严格对齐+补0
gene_template <- data.frame(SYMBOL = gene_order)

if(!require("tidyr")) install.packages("tidyr")
library(tidyr)
############################################################
## 9. 按 mRNA gene list 严格排序 + 缺失基因【补0填充】+ 去重 三合一核心步骤
############################################################
expr_filtered <- gene_template %>% 
  left_join(expr_gene_final, by = "SYMBOL") %>%  # 严格按模板顺序对齐
  mutate(across(where(is.numeric), ~replace_na(.x, 0))) %>% # 缺失基因 全部补0
  group_by(SYMBOL) %>% # 最终兜底去重，防止任何重复基因
  summarise(across(where(is.numeric), mean)) %>% 
  ungroup()

# 安全校验：无重复基因、无NA值
stopifnot(!any(duplicated(expr_filtered$SYMBOL)))
stopifnot(!any(is.na(expr_filtered)))

############################################################
## 10. 保存 RMA 表达矩阵（保留 3 位小数，原代码不变）
############################################################
expr_filtered_round <- expr_filtered
expr_filtered_round[, -1] <- round(expr_filtered_round[, -1], 3)
write.csv(
  expr_filtered_round,
  file = "GSE62254_expression_our.csv",
  row.names = FALSE
)

############################################################
## 11. 计算 Z-score（按基因 / 行，新增NA值补0，避免报错）
############################################################
zscore_matrix <- t(scale(t(expr_filtered[, -1])))
zscore_matrix[is.na(zscore_matrix)] <- 0 # 全0基因计算Z-score会出NA，统一补0
expr_zscore <- cbind(
  SYMBOL = expr_filtered$SYMBOL,
  round(zscore_matrix, 3)
)

############################################################
## 12. 保存 Z-score 版本（原代码不变）
############################################################
write.csv(
  expr_zscore,
  file = "GSE62254_Zscore__expression_our.csv",
  row.names = FALSE
)


