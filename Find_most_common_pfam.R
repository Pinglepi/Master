library(dplyr)
library(stringr)
library(tidyr)

# Function to extract Pfam IDs from hit_list, handling NULL and extracting from characters
extract_pfam_ids <- function(hit_list) {
  pfam_ids <- character(0)  # Initialize an empty vector for Pfam IDs
  for (item in hit_list) {
    if (!is.null(item)) {
      current_ids <- unlist(str_extract_all(item, "PF\\d{5}\\.\\d+"))
      pfam_ids <- c(pfam_ids, current_ids)
    }
  }
  return(pfam_ids)
}
# Initialize an empty data frame to store Pfam counts from each file
pfam_counts <- data.frame(Pfam = character(), Count = integer(), stringsAsFactors = FALSE)

# List all .RData files in the lorf_tbl_pfam folder
files <- list.files(path = "Data/lorf_tbl_pfam", full.names = TRUE, pattern = "\\.RData$")

# Iterate through each file
for (file_path in files) {
  load(file_path)  # Load the .RData file
  
  # Filter lorfs based on amino acid length criteria
  lorf.tbl <- lorf.tbl %>% 
    filter(Length_aa >= 100, Length_aa <= 200)
  
  # Process the data
  file_pfam_counts <- lorf.tbl %>%
    mutate(Pfam_Hits = sapply(hit_list, extract_pfam_ids, simplify = FALSE)) %>%
    unnest(Pfam_Hits) %>%
    group_by(Pfam_Hits) %>%
    summarise(Count = n(), .groups = 'drop')
  
  # Rename the columns appropriately
  colnames(file_pfam_counts) <- c("Pfam", "Count")
  
  # Combine with the main pfam_counts dataframe
  pfam_counts <- bind_rows(pfam_counts, file_pfam_counts)
  
  # Feedback to console
  print(paste0("Processed ", basename(file_path), " with ", nrow(lorf.tbl), " lorfs within length criteria."))
}

# Summarise counts across all files
final_pfam_counts <- pfam_counts %>%
  group_by(Pfam) %>%
  summarise(Total_Count = sum(Count)) %>%
  arrange(desc(Total_Count))

# Print the final table of Pfam domains and their counts
print(final_pfam_counts)

# Optionally, save this data to a CSV file
write.csv(final_pfam_counts, "Data/pfam_counts.csv", row.names = FALSE)
