'Perform seurat integration on the given data
Usage:
    denoise.R <train_x> <train_y> <test_x> <output>
    
Options:
    -h --help  Show this screen.
    -v --version  Show version.

Arguments:
    train_x  folder storing train_X data
    train_y  folder storing train_y data
    test_x  folder storing test_X data
    output  folder to save the output data
' -> doc

suppressMessages(library(docopt))
suppressMessages(library(Seurat))
suppressMessages(library(ggplot2))
suppressMessages(library(patchwork))
suppressMessages(library(dplyr))
suppressMessages(library(Matrix))

load_data <- function(mtx, var, obs) {
  mtx <- Matrix::readMM(mtx)
  mtx <- as(mtx, "dgCMatrix")
  var <- read.csv(var)
  obs <- read.csv(obs)
  colnames(mtx) <- obs[,1]
  rownames(mtx) <- var[,1]
  mtx
}

train_x_folder <- args$train_x
train_y_folder <- args$train_y
test_x_folder <- args$test_x
output_folder <- args$output

train_mtx <- load_data(mtx=file.path(train_x_folder, "x.mtx"),
                       var=file.path(train_x_folder, "var.csv"),
                       obs=file.path(train_x_folder, "obs.csv"))

train_seurat_obj <- CreateSeuratObject(counts=train_mtx,
                                       project="citeseq2")

train_y_mtx <- load_data(mtx=file.path(train_y_folder, "x.mtx"),
                         var=file.path(train_y_folder, "var.csv"),
                         obs=file.path(train_y_folder, "obs.csv"))

adt <- CreateAssayObject(counts=train_y_mtx)
train_seurat_obj[["ADT"]] <- adt
train_seurat_obj@meta.data <- obs
DefaultAssay(train_seurat_obj) <- 'RNA'
train_seurat_obj <- NormalizeData(train_seurat_obj) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

DefaultAssay(train_seurat_obj) <- 'ADT'
VariableFeatures(train_seurat_obj) <- rownames(train_seurat_obj[["ADT"]])
train_seurat_obj <- ScaleData(train_seurat_obj) %>% RunPCA(reduction.name = 'apca')
train_seurat_obj <- FindMultiModalNeighbors(
  train_seurat_obj, reduction.list = list("pca", "apca"), 
  dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
)

DefaultAssay(train_seurat_obj) <- 'RNA'
train_seurat_obj <- RunUMAP(train_seurat_obj, nn.name = "weighted.nn", reduction.name = "wnn.umap", 
                      reduction.key = "wnnUMAP_", 
                      return.model = T)
train_seurat_obj <- FindClusters(train_seurat_obj, graph.name = "wsnn", 
                                 algorithm = 3, resolution = 2, verbose = FALSE)
train_seurat_obj <- RunSPCA(train_seurat_obj, graph = 'wsnn')

query <- load_data(mtx=file.path(test_x_folder, "x.mtx"),
                   var=file.path(test_x_folder, "var.csv"),
                   obs=file.path(test_x_folder, "obs.csv"))

query <- CreateSeuratObject(counts=query,
                            project="citeseq2")

query <- NormalizeData(query)
query <- FindVariableFeatures(query)
query <- ScaleData(query)

query <- RunUMAP(query, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
query <- FindClusters(query, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)
query <- RunSPCA(query, graph = 'wsnn')

anchors <- FindTransferAnchors(
  reference = train_seurat_obj,
  query = query,
  #normalization.method = "SCT",
  reference.reduction = "spca",
  dims = 1:50
)

query <- MapQuery(
  anchorset = anchors,
  query = query,
  reference = train_seurat_obj,
  refdata = list(
    cell_type = "cell_type",
    timepoint = "timepoint",
    predicted_ADT = "ADT"
  ),
  reference.reduction = "spca", 
  reduction.model = "wnn.umap"
)

writeMM(query$predicted_ADT@data, file=file.path(output_folder, "pred_x.mtx"))
write.csv(data.frame(list(var=query$predicted_ADT@data@Dimnames[[1]])), 
          file.path(output_folder, "var.csv"))

write.csv(data.frame(list(obs=query$predicted_ADT@data@Dimnames[[2]])), 
          file.path(output_folder, "obs.csv"))