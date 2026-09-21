#!/usr/bin/env Rscript
# Stage 03 — 16S DADA2 + SILVA (paired amplicon). Independent of RNA-seq.
# Usage: Rscript 03_16s_dada2.R sample_id R1.fastq.gz R2.fastq.gz out_dir
suppressPackageStartupMessages({
  library(dada2)
})
args <- commandArgs(trailingOnly = TRUE)
sample <- args[[1]]; r1 <- args[[2]]; r2 <- args[[3]]; out <- args[[4]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
filt_r1 <- file.path(out, "filt_R1.fastq.gz")
filt_r2 <- file.path(out, "filt_R2.fastq.gz")
filterAndTrim(r1, filt_r1, r2, filt_r2, truncLen = c(240, 200),
              maxEE = c(2, 2), compress = TRUE, multithread = TRUE)
err_f <- learnErrors(filt_r1, multithread = TRUE)
err_r <- learnErrors(filt_r2, multithread = TRUE)
dada_f <- dada(filt_r1, err = err_f, multithread = TRUE)
dada_r <- dada(filt_r2, err = err_r, multithread = TRUE)
merged <- mergePairs(dada_f, filt_r1, dada_r, filt_r2)
seqtab <- makeSequenceTable(list(merged))
seqtab <- removeBimeraDenovo(seqtab, method = "consensus", multithread = TRUE)
# Assign taxonomy with SILVA train set (path from env SILVA_TRAIN)
silva <- Sys.getenv("SILVA_TRAIN")
taxa <- assignTaxonomy(seqtab, silva, multithread = TRUE)
write.csv(t(seqtab), file.path(out, "asv_counts.csv"))
write.csv(taxa, file.path(out, "asv_taxonomy.csv"))
