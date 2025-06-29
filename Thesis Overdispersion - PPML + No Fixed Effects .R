# ===============================
# YOUTUBE THUMBNAIL PANEL ANALYSIS SCRIPT (PPML VERSION) WITHOUT FIXED EFFECTS
# ===============================

# 1. Load Required Libraries
# --------------------------
# Install these if not already installed:
# install.packages(c("tidyverse", "MatchIt", "fixest", "cobalt", "nnet", "broom", "ggeffects", "ggplot2"))

library(tidyverse)
library(MatchIt)
library(fixest)   # For PPML regression
library(cobalt)
library(nnet)     # For multinomial logistic regression
library(broom)    # For tidying model outputs
library(ggeffects)
library(ggplot2)

# 2. Load Data and Preparation
# ----------------------------

file_path <- "C:/Users/ChanK/OneDrive - Tilburg University/Thesis 2024/Youtube Data/Final 2025 June/20250624merged_final_df_v2.csv"
df <- read.csv(file_path)

# Preview the first few rows to check the data
head(df)

# Ensure all relevant variables are the correct type, including new covariates
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
    time_month = format(published_at, "%Y-%m") # Still available for plotting, but not used as FE
  )

# Create the combined treatment variable (four categories)
df <- df %>%
  mutate(
    face_text_combo = case_when(
      face_presence == 1 & text_presence == 1 ~ "both",
      face_presence == 1 & text_presence == 0 ~ "face_only",
      face_presence == 0 & text_presence == 1 ~ "text_only",
      face_presence == 0 & text_presence == 0 ~ "neither"
    ) %>% factor(levels = c("neither", "face_only", "text_only", "both"))
  )

# 3. Overdispersion Diagnostics (Recommended)
# -------------------------------------------
check_overdisp <- function(var) {
  m <- mean(df[[var]], na.rm = TRUE)
  v <- var(df[[var]], na.rm = TRUE)
  cat(sprintf("%s: Mean = %.2f, Variance = %.2f (Ratio = %.2f)\n", 
              var, m, v, v/m))
}
check_overdisp("view_count")
check_overdisp("like_count")

# 4. Propensity Score Matching (for face_presence)
# -----------------------------------------------
covariates <- c("channel_subscribers", "duration", "title_len", "tags_presence",
                "colorfulness_pylette", "brightness_pil", "published_at", 
                "title_sentiment_score", "text_sentiment_score",
                "edge_count", "entropy_mean", "object_count_yolo") # NEW COVARIATES

ps_formula_face <- as.formula(
  paste("face_presence ~", paste(covariates, collapse = " + "))
)

m.out.face <- matchit(ps_formula_face, data = df, method = "nearest", distance = "logit")

# Check covariate balance
summary(m.out.face)
love.plot(m.out.face, binary = "std")

# Get matched dataset for face_presence
matched_df_face <- match.data(m.out.face) %>%
  mutate(time_month = format(published_at, "%Y-%m")) # Still available for plotting

# 5. Propensity Score Matching (for text_presence)
# ------------------------------------------------
ps_formula_text <- as.formula(
  paste("text_presence ~", paste(covariates, collapse = " + "))
)

m.out.text <- matchit(ps_formula_text, data = df, method = "nearest", distance = "logit")
summary(m.out.text)
love.plot(m.out.text, binary = "std")
matched_df_text <- match.data(m.out.text) %>%
  mutate(time_month = format(published_at, "%Y-%m"))

# 6. Propensity Score Matching (for face_text_combo: multinomial)
# ---------------------------------------------------------------
df <- df %>%
  mutate(face_text_combo = relevel(face_text_combo, ref = "neither"))

multi_ps <- multinom(face_text_combo ~ channel_subscribers + duration + title_len +
                       tags_presence + colorfulness_pylette + brightness_pil + 
                       published_at + title_sentiment_score + text_sentiment_score +
                       edge_count + entropy_mean + object_count_yolo,  # NEW COVARIATES
                     data = df, trace = FALSE)

# Predict propensity scores (example: probability for "neither")
df$multi_pscore <- predict(multi_ps, type = "probs")[,1]

# 7. PPML Regression WITHOUT Fixed Effects
# ----------------------------------------

# For face_presence (binary treatment)
ppml_model_views_face <- fepois(
  view_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_face
)
summary(ppml_model_views_face, vcov = "hetero")

ppml_model_likes_face <- fepois(
  like_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_face
)
summary(ppml_model_likes_face, vcov = "hetero")

# For text_presence (binary treatment)
ppml_model_views_text <- fepois(
  view_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_text
)
summary(ppml_model_views_text, vcov = "hetero")

ppml_model_likes_text <- fepois(
  like_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = matched_df_text
)
summary(ppml_model_likes_text, vcov = "hetero")

# For face_text_combo (multinomial treatment)
ppml_model_views_combo <- fepois(
  view_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = df
)
summary(ppml_model_views_combo, vcov = "hetero")

ppml_model_likes_combo <- fepois(
  like_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo,
  data = df
)
summary(ppml_model_likes_combo, vcov = "hetero")

# 8. Optional: Visualize Results
# ------------------------------
# Example: Plot coefficients for face_text_combo
tidy(ppml_model_views_combo) %>%
  filter(str_detect(term, "face_text_combo")) %>%
  ggplot(aes(x = term, y = estimate)) +
  geom_col() +
  labs(title = "Effect of Thumbnail Elements on Views (PPML)",
       x = "Thumbnail Element Combination",
       y = "Estimated Effect (log scale)")

# ===============================
# END OF MAIN SCRIPT
# ===============================

# ===============================
# PREDICTED VALUES AND VISUALIZATION WITH GGPREDICT
# ===============================

# 1. Generate marginal predictions for face_text_combo
pred_combo <- ggpredict(ppml_model_views_combo, terms = "face_text_combo")

# 2. Plot the predictions using ggplot2
ggplot(pred_combo, aes(x = x, y = predicted)) +
  geom_col(fill = "#4682B4") +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  labs(
    title = "Predicted Like Counts by Thumbnail Type",
    x = "Thumbnail Type",
    y = "Predicted Likes"
  ) +
  theme_minimal()


# ===============================
# END OF SCRIPT
# ===============================
