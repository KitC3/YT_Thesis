# ===============================
# YOUTUBE THUMBNAIL ANALYSIS: PPML & NEGBIN INTERACTION MODELS
# ===============================

# 1. Load Required Libraries
library(tidyverse)
library(MatchIt)
library(fixest)   # For PPML and Negative Binomial regression
library(cobalt)
library(nnet)
library(broom)
library(ggeffects)
library(ggplot2)
library(optmatch)

# Set output directory
output_dir <- "<YOUR_OUTPUT_DIRECTORY_HERE>"  # e.g., "C:/path/to/output/directory"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Load and Prepare Data
# -------------------------
file_path <- "<YOUR_CSV_FILE_PATH_HERE>"      # e.g., "C:/path/to/your/file.csv"

df <- df %>%
  mutate(
    face_presence = as.integer(face_present),
    text_presence = as.integer(text_presence),
    channel_name = as.factor(channel_name),
    published_at = as.Date(published_at),
    title_sentiment_score = as.numeric(title_sentiment_score),
    text_sentiment_score = as.numeric(text_sentiment_score),
    edge_count = as.numeric(edge_count),
    entropy_mean = as.numeric(entropy_mean),
    object_count_yolo = as.numeric(object_count_yolo),
    tags_count = as.numeric(tags_count),
    time_month = format(published_at, "%Y-%m")
  ) %>%
  mutate(
    face_text_combo = case_when(
      face_presence == 1 & text_presence == 1 ~ "both",
      face_presence == 1 & text_presence == 0 ~ "face_only",
      face_presence == 0 & text_presence == 1 ~ "text_only",
      face_presence == 0 & text_presence == 0 ~ "neither"
    ) %>% factor(levels = c("neither", "face_only", "text_only", "both"))
  )

# 4. Overdispersion Diagnostics
check_overdisp <- function(var) {
  m <- mean(df[[var]], na.rm = TRUE)
  v <- var(df[[var]], na.rm = TRUE)
  cat(sprintf("%s: Mean = %.2f, Variance = %.2f (Ratio = %.2f)\n", 
              var, m, v, v/m))
}
check_overdisp("view_count")
check_overdisp("like_count")

# 5. Propensity Score Matching
covariates <- c("channel_subscribers", "duration", "title_len",
                "tags_count",
                "colorfulness_pylette", "brightness_pil", "published_at", 
                "title_sentiment_score", "text_sentiment_score",
                "edge_count", "entropy_mean", "object_count_yolo")

ps_formula_face <- as.formula(
  paste("face_presence ~", paste(covariates, collapse = " + "))
)
m.out.face <- matchit(ps_formula_face, data = df, method = "optimal", distance = "logit")
matched_df_face <- match.data(m.out.face) %>%
  mutate(time_month = format(published_at, "%Y-%m"))

ps_formula_text <- as.formula(
  paste("text_presence ~", paste(covariates, collapse = " + "))
)
m.out.text <- matchit(ps_formula_text, data = df, method = "optimal", distance = "logit")
matched_df_text <- match.data(m.out.text) %>%
  mutate(time_month = format(published_at, "%Y-%m"))

# 6. PPML Regression with Interaction
ppml_model_views <- fepois(
  view_count ~ face_presence * text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_face
)

ppml_model_likes <- fepois(
  like_count ~ face_presence * text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_face
)

# 7. Negative Binomial Regression with Interaction
nb_model_views <- fenegbin(
  view_count ~ face_presence * text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_face
)

nb_model_likes <- fenegbin(
  like_count ~ face_presence * text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_face
)

# 8. Export Results: etable with ALL Fit Statistics
etable(
  ppml_model_views, 
  ppml_model_likes, 
  file = file.path(output_dir, "ppml_models_interaction_results.tex"),
  tex = TRUE,
  fitstat = c("cor2", "pr2", "bic", "n", "ll", "aic"),
  dict = c(
    face_presence = "Face Presence", 
    text_presence = "Text Presence",
    `face_presence:text_presence` = "Face × Text",
    title_sentiment_score = "Title Sentiment",
    text_sentiment_score = "Text Sentiment",
    duration = "Duration",
    title_len = "Title Length",
    tags_count = "No. of Tags",
    colorfulness_pylette = "Colorfulness",
    brightness_pil = "Brightness",
    edge_count = "Edge",
    entropy_mean = "Entropy",
    object_count_yolo = "No. of Objects"
  ),
  title = "PPML Regression with Face × Text Interaction"
)

etable(
  nb_model_views, 
  nb_model_likes, 
  file = file.path(output_dir, "negbin_models_interaction_results.tex"),
  tex = TRUE,
  fitstat = c("cor2", "pr2", "bic", "n", "ll", "aic"),
  dict = c(
    face_presence = "Face Presence", 
    text_presence = "Text Presence",
    `face_presence:text_presence` = "Face × Text",
    title_sentiment_score = "Title Sentiment",
    text_sentiment_score = "Text Sentiment",
    duration = "Duration",
    title_len = "Title Length",
    tags_count = "No. of Tags",
    colorfulness_pylette = "Colorfulness",
    brightness_pil = "Brightness",
    edge_count = "Edge",
    entropy_mean = "Entropy",
    object_count_yolo = "No. of Objects"
  ),
  title = "Negative Binomial Regression with Face × Text Interaction"
)

# ===============================
# END OF SCRIPT
# ===============================
