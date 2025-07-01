# ===============================
# COMBINED SCRIPT: Optimal & Nearest Matching with Output Saving
# ===============================
library(tidyverse)
library(MatchIt)
library(cobalt)
library(gridExtra)

# Set output directory
output_dir <- "<YOUR_OUTPUT_DIRECTORY_HERE>"  # e.g., "C:/path/to/output/directory"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Load and Prepare Data
# -------------------------
file_path <- "<YOUR_CSV_FILE_PATH_HERE>"      # e.g., "C:/path/to/your/file.csv"
df <- read.csv(file_path) %>%
  mutate(
    face_presence = as.integer(face_present),
    text_presence = as.integer(text_presence),
    channel_name = as.factor(channel_name),
    published_at = as.Date(published_at),
    title_sentiment_score = as.numeric(title_sentiment_score),
    text_sentiment_score = as.numeric(text_sentiment_score),
    edge_count = as.numeric(edge_count),
    entropy_mean = as.numeric(entropy_mean),
    tags_count = as.numeric(tags_count),
    object_count_yolo = as.numeric(object_count_yolo),
    face_text_combo = factor(case_when(
      face_presence == 1 & text_presence == 1 ~ "both",
      face_presence == 1 & text_presence == 0 ~ "face_only",
      face_presence == 0 & text_presence == 1 ~ "text_only",
      face_presence == 0 & text_presence == 0 ~ "neither"
    ), levels = c("neither", "face_only", "text_only", "both"))
  )

# 2. Define Covariates and Custom Names
# -------------------------------------
covariates <- c(
  "channel_subscribers", "duration", "title_len", "tags_count",
  "colorfulness_pylette", "brightness_pil", "published_at",
  "title_sentiment_score", "text_sentiment_score",
  "edge_count", "entropy_mean", "object_count_yolo"
)

custom_names <- c(
  "channel_subscribers" = "No. of Subscribers",
  "duration" = "Duration",
  "title_len" = "Title Length",
  "tags_count" = "No. of Tags",
  "colorfulness_pylette" = "Colorfulness",
  "brightness_pil" = "Brightness",
  "published_at" = "Published At",
  "title_sentiment_score" = "Title Sentiment",
  "text_sentiment_score" = "Text Sentiment",
  "edge_count" = "Edge",
  "entropy_mean" = "Entropy",
  "object_count_yolo" = "No. of Objects"
)

# 3. Enhanced Balance Assessment Function
# -------------------------------------------------
assess_balance <- function(treatment_var, data, method = "optimal", is_multinomial = FALSE) {
  before_n <- table(data[[treatment_var]])
  
  if (!is_multinomial) {
    formula <- as.formula(paste(treatment_var, "~", paste(covariates, collapse = "+")))
    m.out <- matchit(
      formula, 
      data = data, 
      method = method,
      distance = "logit"
    )
    matched_data <- match.data(m.out)
    after_n <- table(matched_data[[treatment_var]])
    
    bal <- bal.tab(m.out, un = TRUE, thresholds = c(m = 0.1), 
                   var.names = custom_names)
    
    love_plot <- love.plot(
      m.out, 
      binary = "std", 
      thresholds = 0.1, 
      abs = TRUE,
      var.names = custom_names,
      var.order = "unadjusted", 
      sample.names = c("Unmatched", "Matched"),
      title = paste("Love Plot:", treatment_var, "(", method, "Matching)")
    )
  } else {
    after_n <- NULL
    bal <- bal.tab(data[[treatment_var]], data[covariates], 
                   thresholds = c(m = 0.1), var.names = custom_names)
    love_plot <- love.plot(
      bal, 
      thresholds = 0.1, 
      abs = TRUE, 
      var.names = custom_names,
      title = paste("Love Plot:", treatment_var, "(Unmatched Only)")
    )
  }
  
  # Save outputs
  if (!is_multinomial) {
    # Save balance table
    write.csv(
      as.data.frame(bal$Balance),
      file.path(output_dir, paste0("balance_table_", treatment_var, "_", method, ".csv"))
    )
    
    # Save sample sizes
    write.csv(
      data.frame(Group = names(before_n), Before = as.numeric(before_n), After = as.numeric(after_n)),
      file.path(output_dir, paste0("sample_sizes_", treatment_var, "_", method, ".csv"))
    )
    
    # Save love plot
    png(file.path(output_dir, paste0("love_plot_", treatment_var, "_", method, ".png")), 
        width = 1000, height = 700)
    print(love_plot)
    dev.off()
  } else {
    # Save multinomial outputs
    write.csv(
      as.data.frame(bal$Balance),
      file.path(output_dir, paste0("balance_table_", treatment_var, "_unmatched.csv"))
    )
    
    write.csv(
      data.frame(Group = names(before_n), Before = as.numeric(before_n)),
      file.path(output_dir, paste0("sample_sizes_", treatment_var, "_unmatched.csv"))
    )
    
    png(file.path(output_dir, paste0("love_plot_", treatment_var, "_unmatched.png")), 
        width = 1000, height = 700)
    print(love_plot)
    dev.off()
  }
  
  list(
    balance_table = bal,
    love_plot = love_plot,
    sample_size = list(before = before_n, after = after_n)
  )
}

# 4. Generate Reports for Both Methods
# --------------------------------------
# Run both methods for face presence
face_optimal <- assess_balance("face_presence", df, method = "optimal")
face_nearest <- assess_balance("face_presence", df, method = "nearest")

# Run both methods for text presence
text_optimal <- assess_balance("text_presence", df, method = "optimal")
text_nearest <- assess_balance("text_presence", df, method = "nearest")

# 5. Create Comparison Plots
# ---------------------------
# Function to create comparison plots
create_comparison_plot <- function(treatment_var, optimal_res, nearest_res) {
  # Extract balance metrics
  bal_optimal <- as.data.frame(optimal_res$balance_table$Balance)
  bal_nearest <- as.data.frame(nearest_res$balance_table$Balance)
  
  comparison_df <- data.frame(
    Covariate = rownames(bal_optimal),
    Optimal = bal_optimal$Diff.Adj,
    Nearest = bal_nearest$Diff.Adj
  ) %>% 
    pivot_longer(cols = c(Optimal, Nearest), 
                 names_to = "Method", 
                 values_to = "Std_Difference")
  
  p <- ggplot(comparison_df, aes(x = Covariate, y = Std_Difference, fill = Method)) +
    geom_bar(stat = "identity", position = "dodge") +
    geom_hline(yintercept = 0.1, linetype = "dashed", color = "red") +
    geom_hline(yintercept = -0.1, linetype = "dashed", color = "red") +
    labs(title = paste("Standardized Differences Comparison:", treatment_var),
         y = "Standardized Mean Difference (Matched)",
         x = "Covariate") +
    coord_flip() +
    theme_minimal()
  
  # Save plot
  ggsave(file.path(output_dir, paste0("balance_comparison_", treatment_var, ".png")), 
         p, width = 10, height = 8, dpi = 300)
  return(p)
}

# Generate comparison plots
face_comparison_plot <- create_comparison_plot("face_presence", face_optimal, face_nearest)
text_comparison_plot <- create_comparison_plot("text_presence", text_optimal, text_nearest)

# 6. Balance Tables for All Models with Sample Sizes
# --------------------------------------------------
add_sample_sizes <- function(bal_tab, before_n, after_n) {
  bal_df <- as.data.frame(bal_tab$Balance)
  
  sample_rows <- data.frame(
    Covariate = c("Sample size (Control)", "Sample size (Treated)"),
    Diff.Un = c(before_n["0"], before_n["1"]),
    Diff.Adj = c(after_n["0"], after_n["1"]),
    M.Threshold = ""
  )
  colnames(sample_rows) <- colnames(bal_df)
  
  rbind(bal_df, sample_rows)
}

cat("\n=== Balance Tables for All Models ===\n")
models <- c(
  "PPML with Fixed Effects",
  "PPML without Fixed Effects",
  "Negative Binomial with Fixed Effects",
  "Negative Binomial without Fixed Effects"
)

for (model in models) {
  cat("\n", model, ":\n")
  bal_with_samples <- add_sample_sizes(
    face_optimal$balance_table,
    face_optimal$sample_size$before,
    face_optimal$sample_size$after
  )
  print(bal_with_samples)
  
  # Save balance tables
  write.csv(bal_with_samples, 
            file.path(output_dir, paste0("balance_table_", gsub(" ", "_", model), ".csv")))
}

# END OF SCRIPT
