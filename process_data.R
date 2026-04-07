setwd("Your/Working/Directory/Path") 
getwd()


if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install(
  c("affy", "annotate", "hgu133plus2.db"),
  ask = FALSE,
  update = FALSE
)


if(!require("devtools")) install.packages("devtools")
devtools::install_github("waldronlab/HGNChelper", force = TRUE)


library(affy)
library(hgu133plus2.db)
library(dplyr)
library(HGNChelper) 


cel_path <- "/path/to/your/cel/files"
raw_data <- ReadAffy(celfile.path = cel_path)


eset <- rma(raw_data)
expr_matrix <- exprs(eset)
dim(expr_matrix) 


colnames(expr_matrix) <- sub(
  "(GSM[0-9]+).*",
  "\\1",
  colnames(expr_matrix)
)
stopifnot(!any(duplicated(colnames(expr_matrix))))


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


expr_gene_final <- expr_gene %>%
  group_by(SYMBOL) %>%
  summarise(across(where(is.numeric), mean)) %>% 
  ungroup()



gene_names_to_fix <- expr_gene_final$SYMBOL


corrected_gene_res <- checkGeneSymbols(
  x = gene_names_to_fix,
  species = "human", 
  unmapped.as.na = FALSE 
)


modified_genes <- corrected_gene_res[corrected_gene_res$Approved == FALSE, ]

modified_genes <- modified_genes[!is.na(modified_genes$Suggested.Symbol), ]

real_modified <- modified_genes[modified_genes$x != modified_genes$Suggested.Symbol, ]
print(real_modified)


expr_gene_final$SYMBOL <- corrected_gene_res$Suggested.Symbol

expr_gene_final <- expr_gene_final %>% filter(!is.na(SYMBOL))


expr_gene_final <- expr_gene_final %>%
  group_by(SYMBOL) %>%
  summarise(across(where(is.numeric), mean)) %>% 
  ungroup()


gene_list <- read.csv(
  "/home/wangshaoxiang/wsx_R/human_mrna_genes_CORRECTED.csv",
  stringsAsFactors = FALSE
)
gene_order <- gene_list$external_gene_name

gene_template <- data.frame(SYMBOL = gene_order)

if(!require("tidyr")) install.packages("tidyr")
library(tidyr)

expr_filtered <- gene_template %>% 
  left_join(expr_gene_final, by = "SYMBOL") %>% 
  mutate(across(where(is.numeric), ~replace_na(.x, 0))) %>% 
  group_by(SYMBOL) %>% 
  summarise(across(where(is.numeric), mean)) %>% 
  ungroup()


stopifnot(!any(duplicated(expr_filtered$SYMBOL)))
stopifnot(!any(is.na(expr_filtered)))


expr_filtered_round <- expr_filtered
expr_filtered_round[, -1] <- round(expr_filtered_round[, -1], 3)
write.csv(
  expr_filtered_round,
  file = "GSE62254_expression_our.csv",
  row.names = FALSE
)


zscore_matrix <- t(scale(t(expr_filtered[, -1])))
zscore_matrix[is.na(zscore_matrix)] <- 0
expr_zscore <- cbind(
  SYMBOL = expr_filtered$SYMBOL,
  round(zscore_matrix, 3)
)
write.csv(
  expr_zscore,
  file = "GSE62254_Zscore__expression_our.csv",
  row.names = FALSE
)


