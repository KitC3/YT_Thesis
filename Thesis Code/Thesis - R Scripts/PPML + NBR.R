# ===============================
# YOUTUBE THUMBNAIL ANALYSIS SCRIPT (PPML + NEGATIVE BINOMIAL)
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
library(texreg)   # For LaTeX export
library(optmatch) # For optimal matching
library(MASS)     # For glm.nb (if needed)
library(patchwork) # For combining plots

# 2. Load Data and Preparation
# ----------------------------
# Set output directory
output_dir <- "<YOUR_OUTPUT_DIRECTORY_HERE>"  # e.g., "C:/path/to/output/directory"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. Load and Prepare Data
# -------------------------
file_path <- "<YOUR_CSV_FILE_PATH_HERE>"      # e.g., "C:/path/to/your/file.csv"

df <- read.csv(file_path)

# Preview the first few rows to check the data
head(df)

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

# 3. Overdispersion Diagnostics
# -----------------------------
check_overdisp <- function(var) {
  m <- mean(df[[var]], na.rm = TRUE)
  v <- var(df[[var]], na.rm = TRUE)
  cat(sprintf("%s: Mean = %.2f, Variance = %.2f (Ratio = %.2f)\n", 
              var, m, v, v/m))
}
check_overdisp("view_count")
check_overdisp("like_count")

# 4. Propensity Score Matching (face_presence)
# --------------------------------------------
covariates <- c("channel_subscribers", "duration", "title_len",
                "tags_count",
                "colorfulness_pylette", "brightness_pil", "published_at", 
                "title_sentiment_score", "text_sentiment_score",
                "edge_count", "entropy_mean", "object_count_yolo")

ps_formula_face <- as.formula(
  paste("face_presence ~", paste(covariates, collapse = " + "))
)

m.out.face <- matchit(ps_formula_face, data = df, method = "optimal", distance = "logit")

love.plot(
  m.out.face, binary = "std",
  var.names = c(
    channel_subscribers = "No. of Subscribers",
    duration = "Duration",
    title_len = "Title Length",
    tags_count = "No. of Tags",         
    colorfulness_pylette = "Colorfulness",
    brightness_pil = "Brightness",
    published_at = "Published At",
    title_sentiment_score = "Title Sentiment",
    text_sentiment_score = "Text Sentiment",
    edge_count = "Edge",
    entropy_mean = "Entropy",
    object_count_yolo = "No. of Objects"
  )
)
ggsave(file.path(output_dir, "loveplot_face_presence.png"), width = 8, height = 5)

matched_df_face <- match.data(m.out.face) %>%
  mutate(time_month = format(published_at, "%Y-%m"))

# 5. Propensity Score Matching (text_presence)
# --------------------------------------------
ps_formula_text <- as.formula(
  paste("text_presence ~", paste(covariates, collapse = " + "))
)

m.out.text <- matchit(ps_formula_text, data = df, method = "optimal", distance = "logit")
love.plot(
  m.out.text, binary = "std",
  var.names = c(
    channel_subscribers = "No. of Subscribers",
    duration = "Duration",
    title_len = "Title Length",
    tags_count = "No. of Tags",                  
    colorfulness_pylette = "Colorfulness",
    brightness_pil = "Brightness",
    published_at = "Published At",
    title_sentiment_score = "Title Sentiment",
    text_sentiment_score = "Text Sentiment",
    edge_count = "Edge",
    entropy_mean = "Entropy",
    object_count_yolo = "No. of Objects"
  )
)
ggsave(file.path(output_dir, "loveplot_text_presence.png"), width = 8, height = 5)

matched_df_text <- match.data(m.out.text) %>%
  mutate(time_month = format(published_at, "%Y-%m"))

# 6. Propensity Score Matching (face_text_combo: multinomial)
# -----------------------------------------------------------
df <- df %>%
  mutate(face_text_combo = relevel(face_text_combo, ref = "neither"))

multi_ps <- multinom(face_text_combo ~ channel_subscribers + duration + title_len +
                       tags_count +                     
                       colorfulness_pylette + brightness_pil + 
                       published_at + title_sentiment_score + text_sentiment_score +
                       edge_count + entropy_mean + object_count_yolo,
                     data = df, trace = FALSE)

df$multi_pscore <- predict(multi_ps, type = "probs")[,1]

# 7. PPML Regression (fepois)
# ---------------------------
ppml_model_views_face <- fepois(
  view_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len  + tags_count +           
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_face
)
ppml_model_likes_face <- fepois(
  like_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len  + tags_count +           
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_face
)
ppml_model_views_text <- fepois(
  view_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len  + tags_count +           
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_text
)
ppml_model_likes_text <- fepois(
  like_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len  + tags_count +           
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_text
)
ppml_model_views_combo <- fepois(
  view_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len  + tags_count +           
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = df
)
ppml_model_likes_combo <- fepois(
  like_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len  + tags_count +           
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = df
)

# 8. Negative Binomial Regression (fenegbin)
# ------------------------------------------
nb_model_views_face <- fenegbin(
  view_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_face
)
nb_model_likes_face <- fenegbin(
  like_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_face
)
nb_model_views_text <- fenegbin(
  view_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_text
)
nb_model_likes_text <- fenegbin(
  like_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = matched_df_text
)
nb_model_views_combo <- fenegbin(
  view_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = df
)
nb_model_likes_combo <- fenegbin(
  like_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_count +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + time_month,
  data = df
)

# Export PPML models (fepois) to LaTeX
etable(
  list(
    ppml_model_views_face, ppml_model_likes_face, 
    ppml_model_views_text, ppml_model_likes_text, 
    ppml_model_views_combo, ppml_model_likes_combo
  ),
  file = file.path(output_dir, "ppml_models_results.tex"),
  tex = TRUE, # output as LaTeX
  fitstat = ~ . + n + ll + aic, # include N, logLik, AIC
  title = "PPML Regression Results for YouTube Thumbnails"
)

# Export Negative Binomial models (fenegbin) to LaTeX
etable(
  list(
    nb_model_views_face, nb_model_likes_face, 
    nb_model_views_text, nb_model_likes_text, 
    nb_model_views_combo, nb_model_likes_combo
  ),
  file = file.path(output_dir, "negbin_models_results.tex"),
  tex = TRUE, # output as LaTeX
  fitstat = ~ . + n + ll + aic, # include N, logLik, AIC
  title = "Negative Binomial Regression Results for YouTube Thumbnails"
)

# 10. Model Diagnostics
# --------------------
# Residual plots
png(file.path(output_dir, "ppml_residuals.png"), width = 1200, height = 800)
par(mfrow = c(2, 2))
plot(residuals(ppml_model_views_face, type = "deviance"), main = "PPML Residuals: Views (Face)")
plot(residuals(ppml_model_likes_face, type = "deviance"), main = "PPML Residuals: Likes (Face)")
plot(residuals(ppml_model_views_text, type = "deviance"), main = "PPML Residuals: Views (Text)")
plot(residuals(ppml_model_likes_text, type = "deviance"), main = "PPML Residuals: Likes (Text)")
par(mfrow = c(1, 1))
dev.off()

png(file.path(output_dir, "negbin_residuals.png"), width = 1200, height = 800)
par(mfrow = c(2, 2))
plot(residuals(nb_model_views_face, type = "deviance"), main = "NegBin Residuals: Views (Face)")
plot(residuals(nb_model_likes_face, type = "deviance"), main = "NegBin Residuals: Likes (Face)")
plot(residuals(nb_model_views_text, type = "deviance"), main = "NegBin Residuals: Views (Text)")
plot(residuals(nb_model_likes_text, type = "deviance"), main = "NegBin Residuals: Likes (Text)")
par(mfrow = c(1, 1))
dev.off()

# 11. Coefficient Visualizations
# -----------------------------
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
    title = "Effect of Thumbnail Elements on Views (PPML)",
    x = "Covariate",
    y = "Estimated Effect (log scale)"
  ) +
  theme_minimal()
ggsave(file.path(output_dir, "coef_plot_ppml_views.png"), coef_plot_ppml, width = 8, height = 5)

coef_plot_negbin <- tidy(nb_model_views_combo) %>%
  mutate(term_label = recode(term, !!!custom_labels)) %>%
  filter(!is.na(term_label)) %>%
  ggplot(aes(x = term_label, y = estimate)) +
  geom_col() +
  labs(
    title = "Effect of Thumbnail Elements on Views (NegBin)",
    x = "Covariate",
    y = "Estimated Effect (log scale)"
  ) +
  theme_minimal()
ggsave(file.path(output_dir, "coef_plot_negbin_views.png"), coef_plot_negbin, width = 8, height = 5)

# 12. Predicted Value Plots with Fixed Y-Axis Scale
# -------------------------------------------------
# --- Predicted values for views ---
pred_ppml_views <- ggpredict(ppml_model_views_combo, terms = "face_text_combo") %>% as.data.frame()
pred_nb_views   <- ggpredict(nb_model_views_combo,   terms = "face_text_combo") %>% as.data.frame()
all_y_views <- c(pred_ppml_views$conf.low, pred_ppml_views$conf.high, pred_nb_views$conf.low, pred_nb_views$conf.high)
ymin_views <- min(0, floor(min(all_y_views, na.rm = TRUE)))
ymax_views <- ceiling(max(all_y_views, na.rm = TRUE))
ybreaks_views <- pretty(c(ymin_views, ymax_views), n = 5)

# --- Predicted values for likes ---
pred_ppml_likes <- ggpredict(ppml_model_likes_combo, terms = "face_text_combo") %>% as.data.frame()
pred_nb_likes   <- ggpredict(nb_model_likes_combo,   terms = "face_text_combo") %>% as.data.frame()
all_y_likes <- c(pred_ppml_likes$conf.low, pred_ppml_likes$conf.high, pred_nb_likes$conf.low, pred_nb_likes$conf.high)
ymin_likes <- min(0, floor(min(all_y_likes, na.rm = TRUE)))
ymax_likes <- ceiling(max(all_y_likes, na.rm = TRUE))
ybreaks_likes <- pretty(c(ymin_likes, ymax_likes), n = 5)

# --- Plotting function with fixed y-axis scale ---
plot_predicted_fixed <- function(pred, model_name, outcome_label, y_limits, y_breaks) {
  ggplot(pred, aes(x = x, y = predicted)) +
    geom_col(fill = ifelse(model_name == "PPML", "#4682B4", "#DAA520")) +
    geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
    labs(title = paste(model_name, ": Predicted", outcome_label, "by Thumbnail Type"),
         x = "Thumbnail Type", y = paste("Predicted", outcome_label)) +
    scale_y_continuous(limits = y_limits, breaks = y_breaks, expand = c(0,0)) +
    theme_minimal()
}

# --- Create and save plots for views (fixed scale) ---
plot_pred_combo_ppml_views <- plot_predicted_fixed(pred_ppml_views, "PPML", "Views", c(ymin_views, ymax_views), ybreaks_views)
plot_pred_combo_nb_views   <- plot_predicted_fixed(pred_nb_views,   "NB",   "Views", c(ymin_views, ymax_views), ybreaks_views)
pred_combined_views <- plot_pred_combo_ppml_views | plot_pred_combo_nb_views
ggsave(file.path(output_dir, "predicted_views.png"), pred_combined_views, width = 12, height = 6)

# --- Create and save plots for likes (fixed scale) ---
plot_pred_combo_ppml_likes <- plot_predicted_fixed(pred_ppml_likes, "PPML", "Likes", c(ymin_likes, ymax_likes), ybreaks_likes)
plot_pred_combo_nb_likes   <- plot_predicted_fixed(pred_nb_likes,   "NB",   "Likes", c(ymin_likes, ymax_likes), ybreaks_likes)
pred_combined_likes <- plot_pred_combo_ppml_likes | plot_pred_combo_nb_likes
ggsave(file.path(output_dir, "predicted_likes.png"), pred_combined_likes, width = 12, height = 6)

# 13. OPTIONAL: Temporal Trends
# -----------------------------
pred_time_ppml <- ggpredict(
  ppml_model_views_combo,
  terms = c("time_month", "face_text_combo")
)
plot_time_ppml <- ggplot(pred_time_ppml, aes(x = as.Date(paste0(x, "-01")), y = predicted, color = group)) +
  geom_line(size = 1.2) +
  labs(
    title = "Predicted Views Over Time by Thumbnail Type (PPML)",
    x = "Month",
    y = "Predicted Views",
    color = "Thumbnail Type"
  ) +
  theme_minimal()
ggsave(file.path(output_dir, "temporal_trend_ppml.png"), plot_time_ppml, width = 10, height = 6)

pred_time_negbin <- ggpredict(
  nb_model_views_combo,
  terms = c("time_month", "face_text_combo")
)
plot_time_negbin <- ggplot(pred_time_negbin, aes(x = as.Date(paste0(x, "-01")), y = predicted, color = group)) +
  geom_line(size = 1.2) +
  labs(
    title = "Predicted Views Over Time by Thumbnail Type (NegBin)",
    x = "Month",
    y = "Predicted Views",
    color = "Thumbnail Type"
  ) +
  theme_minimal()
ggsave(file.path(output_dir, "temporal_trend_negbin.png"), plot_time_negbin, width = 10, height = 6)

# ===============================
# END OF SCRIPT
# ===============================
