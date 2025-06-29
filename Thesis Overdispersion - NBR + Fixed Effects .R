# ===============================
# YOUTUBE THUMBNAIL PANEL ANALYSIS SCRIPT (NEGATIVE BINOMIAL VERSION)
# Channel and Monthly Fixed Effects
# ===============================

# 1. Load Required Libraries
# --------------------------
# Install these if not already installed:
# install.packages(c("tidyverse", "MatchIt", "fixest", "cobalt", "nnet", "broom", "ggeffects", "ggplot2"))

library(tidyverse)
library(MatchIt)
library(fixest)   # For Negative Binomial regression (fenegbin)
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

# Convert published_at to Date and extract month as a fixed effect
df <- df %>%
  mutate(
    published_at = as.Date(published_at),
    published_month = format(published_at, "%Y-%m")
  )

# Ensure all relevant variables are the correct type, including new covariates
df <- df %>%
  mutate(
    face_presence = as.integer(face_present),
    text_presence = as.integer(text_presence),
    channel_name = as.factor(channel_name),
    published_month = as.factor(published_month),
    title_sentiment_score = as.numeric(title_sentiment_score),
    text_sentiment_score = as.numeric(text_sentiment_score),
    edge_count = as.numeric(edge_count),
    entropy_mean = as.numeric(entropy_mean),
    object_count_yolo = as.numeric(object_count_yolo)
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
matched_df_face <- match.data(m.out.face)

# 5. Propensity Score Matching (for text_presence)
# ------------------------------------------------
ps_formula_text <- as.formula(
  paste("text_presence ~", paste(covariates, collapse = " + "))
)

m.out.text <- matchit(ps_formula_text, data = df, method = "nearest", distance = "logit")
summary(m.out.text)
love.plot(m.out.text, binary = "std")
matched_df_text <- match.data(m.out.text)

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

# 7. NEGATIVE BINOMIAL REGRESSION with Channel & Monthly Fixed Effects
# --------------------------------------------------------------------------

# For face_presence (binary treatment)
nb_model_views_face <- fenegbin(
  view_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + published_month,
  data = matched_df_face
)
summary(nb_model_views_face, vcov = "hetero")

nb_model_likes_face <- fenegbin(
  like_count ~ face_presence + text_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + published_month,
  data = matched_df_face
)
summary(nb_model_likes_face, vcov = "hetero")

# For text_presence (binary treatment)
nb_model_views_text <- fenegbin(
  view_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + published_month,
  data = matched_df_text
)
summary(nb_model_views_text, vcov = "hetero")

nb_model_likes_text <- fenegbin(
  like_count ~ text_presence + face_presence + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + published_month,
  data = matched_df_text
)
summary(nb_model_likes_text, vcov = "hetero")

# For face_text_combo (multinomial treatment)
nb_model_views_combo <- fenegbin(
  view_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + published_month,
  data = df
)
summary(nb_model_views_combo, vcov = "hetero")

nb_model_likes_combo <- fenegbin(
  like_count ~ face_text_combo + title_sentiment_score + text_sentiment_score +
    duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil +
    edge_count + entropy_mean + object_count_yolo | channel_name + published_month,
  data = df
)
summary(nb_model_likes_combo, vcov = "hetero")

# 8. Optional: Visualize Results
# ------------------------------
# Example: Plot coefficients for face_text_combo
tidy(nb_model_views_combo) %>%
  filter(str_detect(term, "face_text_combo")) %>%
  ggplot(aes(x = term, y = estimate)) +
  geom_col() +
  labs(title = "Effect of Thumbnail Elements on Views (Negative Binomial)",
       x = "Thumbnail Element Combination",
       y = "Estimated Effect (log scale)")

# ===============================
# END OF MAIN SCRIPT
# ===============================

# ===============================
# PREDICTED VALUES AND VISUALIZATION WITH GGPREDICT
# ===============================

# 1. Generate marginal predictions for face_text_combo
pred_combo <- ggpredict(nb_model_views_combo, terms = "face_text_combo")

# 2. Plot the predictions using ggplot2
ggplot(pred_combo, aes(x = x, y = predicted)) +
  geom_col(fill = "#4682B4") +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  labs(
    title = "Predicted View Counts by Thumbnail Type",
    x = "Thumbnail Type",
    y = "Predicted Views"
  ) +
  theme_minimal()

# ===============================
# OPTIONAL: VISUALIZE TEMPORAL TRENDS
# ===============================

# Generate marginal predictions of view count over time by thumbnail type
pred_time <- ggpredict(
  nb_model_views_combo,  # Your fitted negative binomial model
  terms = c("published_month", "face_text_combo")  # Use the correct time variable
)

# Plot the predicted views over time for each thumbnail type
ggplot(pred_time, aes(x = as.Date(paste0(x, "-01")), y = predicted, color = group)) +
  geom_line(size = 1.2) +
  labs(
    title = "Predicted Views Over Time by Thumbnail Type",
    x = "Month",
    y = "Predicted Views",
    color = "Thumbnail Type"
  ) +
  theme_minimal()
# ===============================
# END OF PREDICTION VISUALIZATION CODE

# Load required package for panel data analysis
library(plm)

# Convert data to pdata.frame for panel analysis
panel_df <- pdata.frame(df, index = c("channel_name", "published_month"))

# Function to run negative binomial equivalent with plm
run_plm_nb <- function(formula, data, model_type = "within") {
  plm(
    formula,
    data = data,
    model = model_type,
    effect = "twoways",  # Channel and month fixed effects
    index = c("channel_name", "published_month")
  )
}

# 7. PANEL REGRESSION WITH CHANNEL & MONTHLY FIXED EFFECTS
# ---------------------------------------------------------
# For face_text_combo (multinomial treatment)
plm_model_views_combo <- run_plm_nb(
  log(view_count + 1) ~ face_text_combo + title_sentiment_score + 
    text_sentiment_score + duration + title_len + tags_presence +
    colorfulness_pylette + brightness_pil + edge_count + entropy_mean + 
    object_count_yolo,
  data = panel_df
)

# For face_presence (binary treatment)
plm_model_views_face <- run_plm_nb(
  log(view_count + 1) ~ face_presence + text_presence + 
    title_sentiment_score + text_sentiment_score + duration + title_len + 
    tags_presence + colorfulness_pylette + brightness_pil + edge_count + 
    entropy_mean + object_count_yolo,
  data = panel_df
)

# 8. SIGNIFICANCE TESTING OF FIXED EFFECTS
# -----------------------------------------
# Joint significance test for fixed effects
pFtest(plm_model_views_combo, 
       plm(log(view_count + 1) ~ 1, data = panel_df, model = "pooling"))

# Individual model summaries
summary(plm_model_views_combo)
summary(plm_model_views_face)

# 9. VISUALIZE FIXED EFFECTS SIGNIFICANCE
# ---------------------------------------
# Extract fixed effects correctly
channel_effects <- fixef(plm_model_views_combo, effect = "individual")
month_effects <- fixef(plm_model_views_combo, effect = "time")

# Convert to plottable data frames
channel_df <- data.frame(
  channel = names(channel_effects),
  effect = as.numeric(channel_effects)
)

month_df <- data.frame(
  month = names(month_effects),
  effect = as.numeric(month_effects)
)

# Plot channel effects (sorted by effect size)
ggplot(channel_df, aes(x = reorder(channel, effect), y = effect)) +
  geom_col(fill = "#4e79a7") +
  labs(title = "Channel Fixed Effects on View Counts", 
       x = "Channel", y = "Effect Size") +
  coord_flip() +
  theme_minimal()

# Plot monthly effects (time series)
ggplot(month_df, aes(x = as.factor(month), y = effect, group = 1)) +
  geom_line(color = "#e15759", linewidth = 1) +
  geom_point(color = "#e15759", size = 2) +
  labs(title = "Monthly Fixed Effects on View Counts", 
       x = "Month", y = "Effect Size") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
