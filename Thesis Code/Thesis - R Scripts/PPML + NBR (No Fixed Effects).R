# ===============================
# YOUTUBE THUMBNAIL PANEL ANALYSIS SCRIPT (NO FIXED EFFECTS)
# ===============================

# 1. Load Required Libraries
# --------------------------
library(tidyverse)
library(MatchIt)
library(fixest)   # For fepois and fenegbin regression
library(cobalt)
library(nnet)     # For multinomial logistic regression
library(broom)    # For tidying model outputs
library(ggeffects)
library(ggplot2)
library(optmatch) # For optimal matching
library(patchwork) # For plot composition

# Set output directory
output_dir <- "<YOUR_OUTPUT_DIRECTORY_HERE>"  # e.g., "C:/path/to/output/directory"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Load and Prepare Data
# -------------------------
file_path <- "<YOUR_CSV_FILE_PATH_HERE>"      # e.g., "C:/path/to/your/file.csv"
df <- read.csv(file_path)

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
  )

df <- df %>%
  mutate(
    face_text_combo = case_when(
      face_presence == 1 & text_presence == 1 ~ "both",
      face_presence == 1 & text_presence == 0 ~ "face_only",
      face_presence == 0 & text_presence == 1 ~ "text_only",
      face_presence == 0 & text_presence == 0 ~ "neither"
    ) %>% factor(levels = c("neither", "face_only", "text_only", "both"))
  )

# 4. Overdispersion Diagnostics
# -----------------------------
check_overdisp <- function(var) {
  m <- mean(df[[var]], na.rm = TRUE)
  v <- var(df[[var]], na.rm = TRUE)
  cat(sprintf("%s: Mean = %.2f, Variance = %.2f (Ratio = %.2f)\n", 
              var, m, v, v/m))
}
check_overdisp("view_count")
check_overdisp("like_count")

# 5. Propensity Score Matching
# ----------------------------
covariates <- c("channel_subscribers", "duration", "title_len",
                "tags_count",
                "colorfulness_pylette", "brightness_pil", "published_at", 
                "title_sentiment_score", "text_sentiment_score",
                "edge_count", "entropy_mean", "object_count_yolo")

# Face presence
ps_formula_face <- as.formula(
  paste("face_presence ~", paste(covariates, collapse = " + "))
)
m.out.face <- matchit(ps_formula_face, data = df, method = "optimal", distance = "logit")
matched_df_face <- match.data(m.out.face)

# Text presence
ps_formula_text <- as.formula(
  paste("text_presence ~", paste(covariates, collapse = " + "))
)
m.out.text <- matchit(ps_formula_text, data = df, method = "optimal", distance = "logit")
matched_df_text <- match.data(m.out.text)

# Multinomial propensity for face_text_combo
df <- df %>%
  mutate(face_text_combo = relevel(face_text_combo, ref = "neither"))
multi_ps <- multinom(face_text_combo ~ channel_subscribers + duration + title_len +
                       tags_count + colorfulness_pylette + brightness_pil + 
                       published_at + title_sentiment_score + text_sentiment_score +
                       edge_count + entropy_mean + object_count_yolo,
                     data = df, trace = FALSE)
df$multi_pscore <- predict(multi_ps, type = "probs")[,1]

# 6. PPML Regression (NO FIXED EFFECTS)
# -------------------------------------
# Face presence
ppml_model_views_face <- fepois(
  view_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_face
)
ppml_model_likes_face <- fepois(
  like_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_face
)

# Text presence
ppml_model_views_text <- fepois(
  view_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_text
)
ppml_model_likes_text <- fepois(
  like_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_text
)

# Combo
ppml_model_views_combo <- fepois(
  view_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = df
)
ppml_model_likes_combo <- fepois(
  like_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = df
)

# 7. Negative Binomial Regression (NO FIXED EFFECTS)
# --------------------------------------------------
nb_model_views_face <- fenegbin(
  view_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_face
)
nb_model_likes_face <- fenegbin(
  like_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_face
)

nb_model_views_text <- fenegbin(
  view_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_text
)
nb_model_likes_text <- fenegbin(
  like_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_text
)

nb_model_views_combo <- fenegbin(
  view_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = df
)
nb_model_likes_combo <- fenegbin(
  like_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count + colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = df
)

# 8. Export Model Results to LaTeX (etable from fixest)
# -----------------------------------------------------
etable(
  list(
    ppml_model_views_face, ppml_model_likes_face, 
    ppml_model_views_text, ppml_model_likes_text, 
    ppml_model_views_combo, ppml_model_likes_combo
  ),
  file = file.path(output_dir, "ppml_models_results_nofe.tex"),
  tex = TRUE,
  fitstat = ~ . + n + ll + aic,
  title = "PPML Regression Results (No Fixed Effects)"
)
etable(
  list(
    nb_model_views_face, nb_model_likes_face, 
    nb_model_views_text, nb_model_likes_text, 
    nb_model_views_combo, nb_model_likes_combo
  ),
  file = file.path(output_dir, "negbin_models_results_nofe.tex"),
  tex = TRUE,
  fitstat = ~ . + n + ll + aic,
  title = "Negative Binomial Regression Results (No Fixed Effects)"
)

# 9. Visualization: Predicted Value Plots with Fixed Y-Axis Scale
# ---------------------------------------------------------------
# Views
pred_ppml_views <- ggpredict(ppml_model_views_combo, terms = "face_text_combo") %>% as.data.frame()
pred_nb_views   <- ggpredict(nb_model_views_combo,   terms = "face_text_combo") %>% as.data.frame()
all_y_views <- c(pred_ppml_views$conf.low, pred_ppml_views$conf.high, pred_nb_views$conf.low, pred_nb_views$conf.high)
ymin_views <- min(0, floor(min(all_y_views, na.rm = TRUE)))
ymax_views <- ceiling(max(all_y_views, na.rm = TRUE))
ybreaks_views <- pretty(c(ymin_views, ymax_views), n = 5)

# Likes
pred_ppml_likes <- ggpredict(ppml_model_likes_combo, terms = "face_text_combo") %>% as.data.frame()
pred_nb_likes   <- ggpredict(nb_model_likes_combo,   terms = "face_text_combo") %>% as.data.frame()
all_y_likes <- c(pred_ppml_likes$conf.low, pred_ppml_likes$conf.high, pred_nb_likes$conf.low, pred_nb_likes$conf.high)
ymin_likes <- min(0, floor(min(all_y_likes, na.rm = TRUE)))
ymax_likes <- ceiling(max(all_y_likes, na.rm = TRUE))
ybreaks_likes <- pretty(c(ymin_likes, ymax_likes), n = 5)

plot_predicted_fixed <- function(pred, model_name, outcome_label, y_limits, y_breaks) {
  ggplot(pred, aes(x = x, y = predicted)) +
    geom_col(fill = ifelse(model_name == "PPML", "#4682B4", "#DAA520")) +
    geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
    labs(title = paste(model_name, ": Predicted", outcome_label, "by Thumbnail Type"),
         x = "Thumbnail Type", y = paste("Predicted", outcome_label)) +
    scale_y_continuous(limits = y_limits, breaks = y_breaks, expand = c(0,0)) +
    theme_minimal()
}

plot_pred_combo_ppml_views <- plot_predicted_fixed(pred_ppml_views, "PPML", "Views", c(ymin_views, ymax_views), ybreaks_views)
plot_pred_combo_nb_views   <- plot_predicted_fixed(pred_nb_views,   "NB",   "Views", c(ymin_views, ymax_views), ybreaks_views)
pred_combined_views <- plot_pred_combo_ppml_views | plot_pred_combo_nb_views
ggsave(file.path(output_dir, "predicted_views_nofe.png"), pred_combined_views, width = 12, height = 6)

plot_pred_combo_ppml_likes <- plot_predicted_fixed(pred_ppml_likes, "PPML", "Likes", c(ymin_likes, ymax_likes), ybreaks_likes)
plot_pred_combo_nb_likes   <- plot_predicted_fixed(pred_nb_likes,   "NB",   "Likes", c(ymin_likes, ymax_likes), ybreaks_likes)
pred_combined_likes <- plot_pred_combo_ppml_likes | plot_pred_combo_nb_likes
ggsave(file.path(output_dir, "predicted_likes_nofe.png"), pred_combined_likes, width = 12, height = 6)

# 10. (Optional) Coefficient Plots
# --------------------------------
custom_labels <- c(
  channel_subscribers = "No. of Subscribers",
  duration = "Duration",
  title_len = "Title Length",
  tags_presence = "Tags Presence",
  colorfulness_pylette = "Colorfulness",
  brightness_pil = "Brightness",
  published_at = "Published At",
  title_sentiment_score = "Title Sentiment",
  text_sentiment_score = "Text Sentiment",
  edge_count = "Edge",
  entropy_mean = "Entropy",
  object_count_yolo = "No. of Objects"
)

coef_plot_ppml <- tidy(ppml_model_views_combo) %>%
  mutate(term_label = recode(term, !!!custom_labels)) %>%
  filter(!is.na(term_label)) %>%
  ggplot(aes(x = term_label, y = estimate)) +
  geom_col() +
  labs(
    title = "Effect of Thumbnail Elements on Views (PPML, No FE)",
    x = "Covariate",
    y = "Estimated Effect (log scale)"
  ) +
  theme_minimal()
ggsave(file.path(output_dir, "coef_plot_ppml_views_nofe.png"), coef_plot_ppml, width = 8, height = 5)

coef_plot_negbin <- tidy(nb_model_views_combo) %>%
  mutate(term_label = recode(term, !!!custom_labels)) %>%
  filter(!is.na(term_label)) %>%
  ggplot(aes(x = term_label, y = estimate)) +
  geom_col() +
  labs(
    title = "Effect of Thumbnail Elements on Views (NegBin, No FE)",
    x = "Covariate",
    y = "Estimated Effect (log scale)"
  ) +
  theme_minimal()
ggsave(file.path(output_dir, "coef_plot_negbin_views_nofe.png"), coef_plot_negbin, width = 8, height = 5)

# ===============================
# END OF SCRIPT
# ===============================
