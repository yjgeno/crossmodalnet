library(dsb)

# dsb example w/o empty droplets
# specify isotype controls to use in step II 
isotypes = c("MouseIgG1kappaisotype_PROT", "MouseIgG2akappaisotype_PROT", 
             "Mouse IgG2bkIsotype_PROT", "RatIgG2bkIsotype_PROT")
# run ModelNegativeADTnorm to model ambient noise and implement step II
raw.adt.matrix = dsb::cells_citeseq_mtx
raw.adt.matrix[isotypes,]
norm.adt = ModelNegativeADTnorm(cell_protein_matrix = raw.adt.matrix,
                                denoise.counts = TRUE,
                                use.isotype.control = TRUE,
                                isotype.control.name.vec = isotypes
)
par(mfrow = c(2,1));
hist(raw.adt.matrix)
hist(norm.adt)

# my raw protein test set
setwd("./git/Multimodal_22/data/cite_touse/")
library(reticulate)
np <- import('numpy')
count_y_raw <- np$load("cite_test_y_raw.npz")
count_y_raw <- count_y_raw[["arr_0"]]
dim(count_y_raw)
proteins <- read.table("proteins.txt", header=F, stringsAsFactors=F)$V1
cell_id_test <- read.table("cell_id_test.txt", header=F, stringsAsFactors=F)$V1
rownames(count_y_raw) <- cell_id_test
colnames(count_y_raw) <- proteins
count_y_raw <- t(count_y_raw)

# dsb
isotypes = c("Mouse-IgG1", "Mouse-IgG2a", "Mouse-IgG2b", 
             "Rat-IgG2b", "Rat-IgG1", "Rat-IgG2a")
count_y_norm = ModelNegativeADTnorm(cell_protein_matrix = count_y_raw,
                                denoise.counts = TRUE,
                                use.isotype.control = TRUE,
                                isotype.control.name.vec = isotypes
)

# vis results
par(mfrow = c(2,1));
hist(count_y_raw, breaks=50)
hist(count_y_norm, breaks=50)

np$savez("cite_test_y_norm.npz", count_y_norm)
