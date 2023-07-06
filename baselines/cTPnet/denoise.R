'Perform SAVERX on th given data
Usage:
    denoise.R <input> <output>
    
Options:
    -h --help  Show this screen.
    -v --version  Show version.

Arguments:
    input  inpur data folders location
    output  folder to save the output data
' -> doc

suppressMessages(library(docopt))
suppressMessages(library(Matrix))
library(SAVERX)

print("loading mtx")

args <- docopt(doc)

input_folder <- args$input
output_folder <- args$output

mtx <- Matrix::readMM(file.path(input_folder, "x.mtx"))
mtx <- as(mtx, "dgCMatrix")
var <- read.csv(file.path(input_folder, "var.csv"))
obs <- read.csv(file.path(input_folder, "obs.csv"))
colnames(mtx) <- obs[,1]
rownames(mtx) <- var[,1]

saveRDS(mtx, file.path(output_folder, "tmp.rds"))
file <- saverx(file.path(output_folder, "tmp.rds"), 
               data.species = "Human", 
               use.pretrain = T, 
               pretrained.weights.file = file.path(input_folder, "human_Immune.hdf5"), 
               model.species = "Human")
denoised_result <- readRDS(file)
write.csv(rownames(denoised_result$estimate), 
        file = file.path(output_folder, "denoised_var.csv"))
write.csv(colnames(denoised_result$estimate), file = 
        file.path(output_folder, "denoised_obs.csv"))
writeMM(obj = Matrix(denoised_result$estimate, sparse = TRUE), 
file = file.path(output_folder, "denoised_mtx.csv"))