library(tidyverse) 
library(microseq)
library(micropan)


load("representative.tbl.RData")

most_common_orders <- representative.tbl %>%
  group_by(order) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

# Define a list of order names to iterate over
order_names <- c("Corynebacteriales", "Enterobacterales", "Lactobacillales", "Burkholderiales")

# Loop through each order name in the list
for(order_name in order_names) {
  # Filter the representative table for the current order
  selected.tbl <- representative.tbl %>% 
    filter(order == order_name)
  
  # Initialize lorfs.tbl for the current order
  lorfs.tbl <- NULL
  
  # Loop through each row of the filtered table
  for(i in 1:nrow(selected.tbl)){
    # Read and process HMM table
    hmm.tbl <- readHmmer(file.path("hmmscan",
                                   str_c(selected.tbl$genome_id[i], ".txt.gz"))) %>% 
      mutate(Query = str_remove_all(Query, "Seqid=|Start=|End=|Strand=")) %>% 
      group_by(Query) %>% 
      slice(1)
    
    # Read, process, and append lORFs data
    lorfs.tbl <- readFasta(file.path("lorf_faa", 
                                     str_c(selected.tbl$genome_id[i], "_lorfs.faa.gz"))) %>%
      mutate(length = str_length(Sequence)) %>% 
      mutate(genome_id = selected.tbl$genome_id[i]) %>% 
      mutate(has_hmm = if_else(Header %in% hmm.tbl$Query, 1, 0)) %>% 
      mutate(GC = selected.tbl$GC[i]) %>%
      bind_rows(lorfs.tbl)
  }
  
  # Save lorfs.tbl as a file named after the order_name
  # Adjust the path and file format as needed
  write.csv(lorfs.tbl, file.path(paste0(order_name, "_lorfs.csv")), row.names = FALSE)
}

# This will create a separate CSV file for each order in the "output" directory
