library(tidyverse)
library(microseq)
library(micropan)
library(dplyr)

##################################
### From the representative.tbl
### we select some genomes
###
load("Data/representative.tbl.RData")

# Required library
library(dplyr)

# Desired total sample size
total_samples <- 250

# Calculate number of entries per class (attempting to distribute them evenly)
samples_per_class <- ceiling(total_samples / length(unique(representative.tbl$class)))

# Sample from each class
selected.tbl <- representative.tbl %>%
  group_by(class) %>%
  sample_n(min(n(), samples_per_class)) %>%
  ungroup()

# If the number of samples is more than required due to ceiling, randomly select required number
if (nrow(selected.tbl) > total_samples) {
  set.seed(123)  # for reproducibility
  selected <- selected.tbl %>% sample_n(total_samples)
}

# Calculate the sum of 'n_lorfs' column in the selected data
sum_n_lorfs <- sum(selected.tbl$n_lorfs, na.rm = TRUE)

#############################################
### LORF data and their response (has_hmm)
###
lorfs.tbl <- NULL
for(i in 1:nrow(selected.tbl)){
  hmm.tbl <- readHmmer(file.path("Data/hmmscan",
                                 str_c(selected.tbl$genome_id[i], ".txt.gz"))) %>% 
    mutate(Query = str_remove_all(Query, "Seqid=|Start=|End=|Strand=")) %>% 
    group_by(Query) %>% 
    slice(1)
  lorfs.tbl <- readFasta(file.path("Data/lorf_faa", 
                                   str_c(selected.tbl$genome_id[i], "_lorfs.faa.gz"))) %>%
    mutate(length = str_length(Sequence)) %>% 
    mutate(genome_id = selected.tbl$genome_id[i]) %>% 
    mutate(has_hmm = if_else(Header %in% hmm.tbl$Query, 1, 0)) %>% 
    mutate(GC = selected.tbl$GC[i]) %>%
    bind_rows(lorfs.tbl)
}
write.csv(lorfs.tbl, "Data/General_lorfs.csv", row.names = FALSE)
