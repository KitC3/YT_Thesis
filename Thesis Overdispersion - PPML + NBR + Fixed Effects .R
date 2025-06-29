# ===============================
# YOUTUBE THUMBNAIL PANEL ANALYSIS: PPML & NEGATIVE BINOMIAL
# ===============================

# 1. Load Required Libraries
library(tidyverse)
library(MatchIt)
library(fixest)
library(broom)
library(ggeffects)
library(ggplot2)
library(patchwork)

# 2. Load and Prepare Data
df <- read.csv("C:/Users/ChanK/OneDrive - Tilburg University/Thesis 2024/Youtube Data/Final 2025 June/20250624merged_final_df_v2.csv") %>%
  mutate(
    face_presence = as.integer(face_present),
    text_presence = as.integer(text_presence),
    channel_name = as.factor(channel_name),
    published_at = as.Date(published_at),
    published_month = format(published_at, "%Y-%m"),
    title_sentiment_score = as.numeric(title_sentiment_score),
    text_sentiment_score = as.numeric(text_sentiment_score),
    edge_count = as.numeric(edge_count),
    entropy_mean = as.numeric(entropy_mean),
    object_count_yolo = as.numeric(object_count_yolo),
    face_text_combo = case_when(
      face_presence == 1 & text_presence == 1 ~ "both",
      face_presence == 1 & text_presence == 0 ~ "face_only",
      face_presence == 0 & text_presence == 1 ~ "text_only",
      face_presence == 0 & text_presence == 0 ~ "neither"
    ) %>% factor(levels = c("neither", "face_only", "text_only", "both"))
  )

# 3. Propensity Score Matching
covariates <- c("channel_subscribers", "duration", "title_len", "tags_presence",
                "colorfulness_pylette", "brightness_pil", "published_at",
                "title_sentiment_score", "text_sentiment_score",
                "edge_count", "entropy_mean", "object_count_yolo")

# Face treatment matching
ps_formula_face <- as.formula(paste("face_presence ~", paste(covariates, collapse = " + ")))
m.out.face <- matchit(ps_formula_face, data = df, method = "nearest", distance = "logit")
matched_df_face <- match.data(m.out.face)

# Text treatment matching
ps_formula_text <- as.formula(paste("text_presence ~", paste(covariates, collapse = " + ")))
m.out.text <- matchit(ps_formula_text, data = df, method = "nearest", distance = "logit")
matched_df_text <- match.data(m.out.text)

# 4. Run PPML and NB Models
# Face treatment models
ppml_model_views_face <- fepois(view_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
                                  duration + title_len + tags_presence + colorfulness_pylette + brightness_pil +
                                  edge_count + entropy_mean + object_count_yolo | channel_name + published_month, data = matched_df_face)
ppml_model_likes_face <- update(ppml_model_views_face, like_count ~ .)

nb_model_views_face <- fenegbin(view_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
                                  duration + title_len + tags_presence + colorfulness_pylette + brightness_pil +
                                  edge_count + entropy_mean + object_count_yolo | channel_name + published_month, data = matched_df_face)
nb_model_likes_face <- update(nb_model_views_face, like_count ~ .)

# Text treatment models
ppml_model_views_text <- fepois(view_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
                                  duration + title_len + tags_presence + colorfulness_pylette + brightness_pil +
                                  edge_count + entropy_mean + object_count_yolo | channel_name + published_month, data = matched_df_text)
ppml_model_likes_text <- update(ppml_model_views_text, like_count ~ .)

nb_model_views_text <- fenegbin(view_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
                                  duration + title_len + tags_presence + colorfulness_pylette + brightness_pil +
                                  edge_count + entropy_mean + object_count_yolo | channel_name + published_month, data = matched_df_text)
nb_model_likes_text <- update(nb_model_views_text, like_count ~ .)

# Combo treatment models
ppml_model_views_combo <- fepois(view_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
                                   duration + title_len + tags_presence + colorfulness_pylette + brightness_pil +
                                   edge_count + entropy_mean + object_count_yolo | channel_name + published_month, data = df)
ppml_model_likes_combo <- update(ppml_model_views_combo, like_count ~ .)

nb_model_views_combo <- fenegbin(view_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
                                   duration + title_len + tags_presence + colorfulness_pylette + brightness_pil +
                                   edge_count + entropy_mean + object_count_yolo | channel_name + published_month, data = df)
nb_model_likes_combo <- update(nb_model_views_combo, like_count ~ .)

# 5. Create Summary Table
tidy_model <- function(model, model_name, outcome) {
  tidy(model, vcov = "hetero") %>%
    mutate(model = model_name, outcome = outcome) %>%
    select(model, outcome, term, estimate, std.error, statistic, p.value)
}

all_results <- bind_rows(
  tidy_model(ppml_model_views_face, "PPML_face", "view_count"),
  tidy_model(ppml_model_likes_face, "PPML_face", "like_count"),
  tidy_model(ppml_model_views_text, "PPML_text", "view_count"),
  tidy_model(ppml_model_likes_text, "PPML_text", "like_count"),
  tidy_model(ppml_model_views_combo, "PPML_combo", "view_count"),
  tidy_model(ppml_model_likes_combo, "PPML_combo", "like_count"),
  tidy_model(nb_model_views_face, "NB_face", "view_count"),
  tidy_model(nb_model_likes_face, "NB_face", "like_count"),
  tidy_model(nb_model_views_text, "NB_text", "view_count"),
  tidy_model(nb_model_likes_text, "NB_text", "like_count"),
  tidy_model(nb_model_views_combo, "NB_combo", "view_count"),
  tidy_model(nb_model_likes_combo, "NB_combo", "like_count")
)

write.csv(all_results, "model_summary_table.csv", row.names = FALSE)

# 6. Visualization: Forest Plots
plot_forest <- function(results_df, model_pattern, outcome_pattern) {
  results_df %>%
    filter(str_detect(model, model_pattern), 
           outcome == outcome_pattern, 
           !term %in% c("(Intercept)", ".theta")) %>%
    ggplot(aes(x = reorder(term, estimate), y = estimate, color = model)) +
    geom_point(position = position_dodge(width = 0.5)) +
    geom_errorbar(aes(ymin = estimate - 1.96 * std.error, 
                      ymax = estimate + 1.96 * std.error),
                  width = 0.2, position = position_dodge(width = 0.5)) +
    coord_flip() +
    labs(title = paste("Forest Plot:", model_pattern, outcome_pattern),
         x = "Predictor", y = "Estimate (log scale)") +
    theme_minimal()
}

# 7. Visualization: Predicted Value Plots with Fixed Y-Axis Scale

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
ggsave("predicted_views.png", pred_combined_views, width = 12, height = 6)

# --- Create and save plots for likes (fixed scale) ---
plot_pred_combo_ppml_likes <- plot_predicted_fixed(pred_ppml_likes, "PPML", "Likes", c(ymin_likes, ymax_likes), ybreaks_likes)
plot_pred_combo_nb_likes   <- plot_predicted_fixed(pred_nb_likes,   "NB",   "Likes", c(ymin_likes, ymax_likes), ybreaks_likes)
pred_combined_likes <- plot_pred_combo_ppml_likes | plot_pred_combo_nb_likes
ggsave("predicted_likes.png", pred_combined_likes, width = 12, height = 6)

# --- Forest plots for all models/outcomes ---
forest_face_view <- plot_forest(all_results, "face", "view_count")
forest_face_like <- plot_forest(all_results, "face", "like_count")
forest_text_view <- plot_forest(all_results, "text", "view_count")
forest_text_like <- plot_forest(all_results, "text", "like_count")
forest_combo_view <- plot_forest(all_results, "combo", "view_count")
forest_combo_like <- plot_forest(all_results, "combo", "like_count")

forest_combined <- (forest_face_view | forest_face_like) / 
  (forest_text_view | forest_text_like) / 
  (forest_combo_view | forest_combo_like)
ggsave("forest_plots.png", forest_combined, width = 16, height = 12)

# 8. Diagnostic Plots (Residuals)
plot_residuals <- function(model, model_name) {
  residuals <- tibble(
    fitted = fitted(model),
    resid = residuals(model, type = "response")
  )
  ggplot(residuals, aes(x = fitted, y = resid)) +
    geom_point(alpha = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = paste(model_name, ": Residuals vs Fitted"),
         x = "Fitted Values", y = "Residuals") +
    theme_minimal()
}

resid_ppml <- plot_residuals(ppml_model_views_face, "PPML")
resid_nb <- plot_residuals(nb_model_views_face, "NB")
resid_combined <- resid_ppml | resid_nb
ggsave("residual_plots.png", resid_combined, width = 12, height = 6)

# ===============================
# END OF SCRIPT
# ===============================
