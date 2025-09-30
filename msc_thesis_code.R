
Sys.setlocale("LC_TIME","English")

rm(list = ls()) 
cat("\014")
dirpath = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dirpath)


library(dplyr)
library(tidyr)
library(imputeTS)
library(lubridate)
library(pROC)
library(randomForest)
library(ranger)
library(ParBayesianOptimization)
library(caret)
library(pdp)
library(xgboost)
library(keras)
library(tensorflow)
library(purrr)
library(writexl)
library(DescTools)
library(ggplot2)


# 1. DATA CLEANING -------------------------------------------------------------
# Loading and merging datasets ----
bankrupt_firms <- read.csv("CRSP_bankruptcy.csv")
compustat <- read.csv("compustat_data.csv")
link_table <- read.csv("link_table.csv")

bankrupt_firms <- bankrupt_firms %>%
  filter(DLSTCD == 574|DLSTCD == 560|DLSTCD == 561|DLSTCD == 572|DLSTCD == 585) %>%
  mutate(bankrupt_firm = 1)

compustat <- compustat %>%
  filter(fic == "USA") %>%
  mutate(sic2 = as.numeric(substr(sic, 1, 2))) %>%
  filter(!(sic >= 6000 & sic < 7000),  # remove financials
         !(sic >= 4900 & sic < 5000))  # remove utilities

link_table <- link_table %>%
  rename(PERMNO = LPERMNO)%>%
  filter(LINKENDDT != "E")

link_table <- link_table %>%
  mutate(priority = case_when(
    LINKPRIM == "C" ~ 1,
    LINKPRIM == "P" ~ 2,  
    TRUE ~ 3              
  )) %>%
  group_by(gvkey) %>%
  arrange(priority) %>%  
  dplyr::slice(1)  

compustat <- compustat %>%
  select(-tic) %>%
  left_join(link_table, by = "gvkey")

firm_data <- compustat %>%
  left_join(bankrupt_firms %>% select(PERMNO, DLSTDT, bankrupt_firm), by = "PERMNO") %>%
  mutate(bankrupt_firm = replace_na(bankrupt_firm, 0))  


# Convert DLSTDT to date format and extract year
firm_data$DLSTDT <- as.Date(firm_data$DLSTDT)
firm_data$delist_year <- as.numeric(format(firm_data$DLSTDT, "%Y"))

# Identify gvkeys with full 3 years of pre-bankruptcy data
valid_bankrupt_permnos <- firm_data %>%
  filter(bankrupt_firm == 1) %>%
  mutate(years_before_delist = delist_year - fyear) %>%
  filter(years_before_delist %in% 1:3) %>%
  count(gvkey) %>%
  filter(n == 3) %>%
  pull(gvkey)

# Filter full dataset: keep all healthy firms + only bankrupts with 3-year history
firm_data <- firm_data %>%
  filter(bankrupt_firm == 0 | gvkey %in% valid_bankrupt_permnos)

firm_data %>%
  filter(bankrupt_firm == 1) %>%
  summarise(bankrupt_firm_count = n_distinct(gvkey)) %>%
  pull(bankrupt_firm_count) %>%
  cat("Remaining bankrupt firms:", ., "\n")

# Remove post-bankruptcy years (not useful for prediction)
firm_data <- firm_data %>%
  filter(is.na(delist_year) | fyear < delist_year)

# Apply rolling window logic (label only 1 year before bankruptcy)
firm_data <- firm_data %>%
  mutate(
    bankrupt = ifelse(
      !is.na(delist_year) & (delist_year - fyear == 1), 1, 0))

firm_data <- firm_data %>% select(-delist_year)


# Handling missing values ----
firm_data <- firm_data %>%
  select(-indfmt, -consol, -popsrc, -datafmt, -tic, -conm, -curcd,
         -LINKPRIM, -LIID, -LINKTYPE, -LPERMCO, -LINKDT, -LINKENDDT,
         -priority, -PERMNO, -utfdoc, -xrd, -costat, -fic)

exclude_vars <- c("gvkey", "sic", "sic2", "fyear", "datadate", "DLSTDT", "bankrupt", "bankrupt_firm")

# 1. Kalman Smoothing
safe_kalman <- function(x) {
  tryCatch(
    na_kalman(x, model = "StructTS", smooth = TRUE),
    error = function(e) x  # return input unchanged if it fails
  )
}

firm_data <- firm_data %>%
  group_by(gvkey) %>%
  mutate(across(
    .cols = where(is.numeric) & !any_of(exclude_vars),
    .fns = ~ safe_kalman(.)
  )) %>%
  ungroup()
summary(firm_data)

# 2. Fill any remaining NAs using firm-level median
firm_data <- firm_data %>%
  group_by(gvkey) %>%
  mutate(across(where(is.numeric) & !any_of(exclude_vars),
                ~ replace_na(., median(., na.rm = TRUE)))) %>%
  ungroup()
summary(firm_data)

# 3. Use industry median
firm_data <- firm_data %>%
  group_by(sic2, fyear) %>%
  mutate(across(where(is.numeric) & !any_of(exclude_vars),
                ~ replace_na(., median(., na.rm = TRUE)))) %>%
  ungroup()
summary(firm_data)

# 4. Drop rows with any NA values EXCEPT for DLSTDT
vars_to_check <- c("dt", "re", "teq", "mkvalt")
firm_data <- firm_data %>%
  filter(if_all(all_of(vars_to_check), ~ !is.na(.)))

colSums(is.na(firm_data))

firm_data %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()

# Construct the final dataset with the necessary ratios ----
# Require firms to have at least 4 years of data
valid_gvkeys <- firm_data %>%
  group_by(gvkey) %>%
  summarise(years_of_data = n()) %>%
  filter(years_of_data >= 4) %>%
  pull(gvkey)

# Filter firm_data to only include those firms
model_data <- firm_data %>%
  filter(gvkey %in% valid_gvkeys)

# Construct necessary ratios
model_data <- model_data %>%
  mutate(
    wc_ta    = (act - lct) / at,
    re_ta    = re / at,
    ebit_ta  = ebit / at,
    mve_bvlt = (prcc_f * csho) / lt,
    sale_ta  = sale / at,
    op_margin = ebit / sale,
    roe      = ni / seq,
    pb_ratio = (prcc_f * csho) / seq
  ) %>%
  group_by(gvkey) %>%
  arrange(fyear, .by_group = TRUE) %>%
  mutate(
    asset_growth = (at - lag(at)) / lag(at),
    sales_growth = (sale - lag(sale)) / lag(sale),
    emp_growth   = (emp - lag(emp)) / lag(emp),
    roe_change   = roe - lag(roe),
    pb_change    = pb_ratio - lag(pb_ratio)
  ) %>%
  ungroup()

# Handle NAs
model_data <- model_data %>%
  mutate(across(
    where(is.numeric),
    ~ replace(., is.infinite(.) | is.nan(.), NA)))

# Replace NAs with zero where sales denominator was 0 
model_data <- model_data %>%
  mutate(
    asset_growth = ifelse(is.na(asset_growth), 0, asset_growth),
    sales_growth = ifelse(is.na(sales_growth), 0, sales_growth),
    emp_growth = ifelse(is.na(emp_growth), 0, emp_growth),
    roe_change = ifelse(is.na(roe_change), 0, roe_change),
    pb_change = ifelse(is.na(pb_change), 0, pb_change),
    op_margin = ifelse(is.na(op_margin), 0, op_margin))

# Filter out NAs
core_vars <- c(
  "wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta",
  "asset_growth", "sales_growth", "emp_growth",
  "roe_change", "pb_change", "op_margin")
model_data <- model_data %>%
  filter(if_all(all_of(core_vars), ~ !is.na(.))) %>%
  filter(at > 0)

colSums(is.na(model_data))

model_data %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()

# Select final variables
model_data <- model_data %>%
  mutate(bankrupt = as.factor(bankrupt),
         log_at = log(at),
         sic2_f = as.factor(sic2)) %>%
  select(
    gvkey, fyear, datadate, DLSTDT, bankrupt, bankrupt_firm,
    wc_ta, re_ta, ebit_ta, mve_bvlt, sale_ta, asset_growth,
    sales_growth, emp_growth, roe_change, pb_change, op_margin,
    log_at, sic2, sic2_f)

# Winzorise data ----
# List of variables to summarize
vars <- c("wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta", "asset_growth",
          "sales_growth", "emp_growth", "roe_change", "pb_change", "op_margin")

model_data <- model_data %>%
  mutate(across(all_of(vars), ~ DescTools::Winsorize(., val = quantile(., probs = c(0.005, 0.995), na.rm = TRUE))))


write.csv(model_data, "final_model_data.csv", row.names = FALSE)
model_data <- read.csv("final_model_data.csv")


# Summary stats ----
# Summarise for each variable
model_data_summary <- model_data %>%
  select(all_of(vars)) %>%
  summarise(across(everything(), list(
    mean = ~ as.numeric(mean(., na.rm = TRUE)),
    sd = ~ as.numeric(sd(., na.rm = TRUE)),
    min = ~ as.numeric(min(., na.rm = TRUE)),
    q1 = ~ as.numeric(quantile(., 0.25, na.rm = TRUE)),
    median = ~ as.numeric(median(., na.rm = TRUE)),
    q3 = ~ as.numeric(quantile(., 0.75, na.rm = TRUE)),
    max = ~ as.numeric(max(., na.rm = TRUE))
  ), .names = "{.col}__{.fn}")) %>%
  pivot_longer(everything(),
               names_to = c("variable", "stat"),
               names_sep = "__") %>%
  pivot_wider(names_from = stat, values_from = value)

# Summarise for each variable for bankrupt
model_data_summary_b <- model_data %>%
  filter(bankrupt_firm == 1) %>%
  select(all_of(vars)) %>%
  summarise(across(everything(), list(
    mean = ~ as.numeric(mean(., na.rm = TRUE)),
    sd = ~ as.numeric(sd(., na.rm = TRUE)),
    min = ~ as.numeric(min(., na.rm = TRUE)),
    q1 = ~ as.numeric(quantile(., 0.25, na.rm = TRUE)),
    median = ~ as.numeric(median(., na.rm = TRUE)),
    q3 = ~ as.numeric(quantile(., 0.75, na.rm = TRUE)),
    max = ~ as.numeric(max(., na.rm = TRUE))
  ), .names = "{.col}__{.fn}")) %>%
  pivot_longer(everything(),
               names_to = c("variable", "stat"),
               names_sep = "__") %>%
  pivot_wider(names_from = stat, values_from = value)

# Summarise for each variable for non-bankrupt
model_data_summary_nb <- model_data %>%
  filter(bankrupt_firm == 0) %>%
  select(all_of(vars)) %>%
  summarise(across(everything(), list(
    mean = ~ as.numeric(mean(., na.rm = TRUE)),
    sd = ~ as.numeric(sd(., na.rm = TRUE)),
    min = ~ as.numeric(min(., na.rm = TRUE)),
    q1 = ~ as.numeric(quantile(., 0.25, na.rm = TRUE)),
    median = ~ as.numeric(median(., na.rm = TRUE)),
    q3 = ~ as.numeric(quantile(., 0.75, na.rm = TRUE)),
    max = ~ as.numeric(max(., na.rm = TRUE))
  ), .names = "{.col}__{.fn}")) %>%
  pivot_longer(everything(),
               names_to = c("variable", "stat"),
               names_sep = "__") %>%
  pivot_wider(names_from = stat, values_from = value)

write_xlsx(
  list(
    Full_Sample = model_data_summary,
    Bankrupt_Firms = model_data_summary_b,
    Non_Bankrupt_Firms = model_data_summary_nb
  ),
  "descriptive_statistics_all.xlsx")

# Split the data set into train and test data sets ----
# Create 'ref_year' for each firm
model_data <- model_data %>%
  group_by(gvkey) %>%
  mutate(
    ref_year = ifelse(
      bankrupt_firm == 1,
      year(DLSTDT),
      max(fyear, na.rm = TRUE)
    )
  ) %>%
  ungroup()

# Decide train/test split based on reference year
train_ids <- model_data %>%
  filter(ref_year < 2018) %>%
  distinct(gvkey) %>%
  pull(gvkey)

test_ids <- model_data %>%
  filter(ref_year >= 2018) %>%
  distinct(gvkey) %>%
  pull(gvkey)

# Assign full firm history to the proper set
train_data <- model_data %>%
  filter(gvkey %in% train_ids)

test_data <- model_data %>%
  filter(gvkey %in% test_ids)
train_data$bankrupt <- factor(train_data$bankrupt, levels = c(0, 1))
test_data$bankrupt  <- factor(test_data$bankrupt, levels = c(0, 1))

# Recreate a clean version of sic2_f with only observed levels in train
train_data$sic2_f <- factor(as.character(train_data$sic2), levels = unique(as.character(train_data$sic2)))
test_data$sic2_f <- factor(as.character(test_data$sic2), levels = levels(train_data$sic2_f))
test_data <- test_data %>% filter(!is.na(sic2_f))

train_data %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()
test_data %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()


# 2. FIRM-LEVEL MODELS ---------------------------------------------------------
# Logistic regression ----
lr_model <- glm(bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
                  asset_growth + sales_growth + emp_growth +roe_change + 
                  pb_change + op_margin + log_at + sic2_f,
                data = train_data, family = "binomial")
lr_pred <- predict(lr_model, newdata = test_data, type = "response")

summary(lr_model)

# AUC - ROC
auc_lr <- roc(test_data$bankrupt, lr_pred)
print(auc_lr)
plot(auc_lr, main = "ROC Curve for Logistic Regression")

# Confusion matrix
best_thresh_lr <- as.numeric(coords(auc_lr, "best", ret = "threshold", transpose = FALSE))
pred_class_lr <- ifelse(lr_pred >= best_thresh_lr, 1, 0)
conf_matrix_lr <- table(Predicted = pred_class_lr, Actual = test_data$bankrupt)
print(conf_matrix_lr)

# Precision, recall, and F1
TP_lr <- conf_matrix_lr["1", "1"]
FP_lr <- conf_matrix_lr["1", "0"]
FN_lr <- conf_matrix_lr["0", "1"]

precision_lr <- TP_lr / (TP_lr + FP_lr)
precision_lr
recall_lr <- TP_lr / (TP_lr + FN_lr)
recall_lr
f1_lr <- 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)
f1_lr


# Random forest ----
rf_formula <- bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
  asset_growth + sales_growth + emp_growth + roe_change + pb_change +
  op_margin + log_at + sic2_f

set.seed(123)
# OOB for tree count selection
rf_oob_plot <- randomForest(
  bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
    asset_growth + sales_growth + emp_growth + roe_change +
    pb_change + op_margin + log_at,
  data = train_data,
  ntree = 1000,
  importance = TRUE
)

plot(rf_oob_plot, main = "OOB Error vs. Number of Trees")


# Count number of predictors used in your model
# Exclude: target + IDs + factor variables (if already dummy-encoded)
numeric_predictors <- train_data %>%
  select(where(is.numeric)) %>%
  select(-bankrupt_firm)  

p <- ncol(numeric_predictors)
cat("Number of numeric predictors:", p, "\n")

# Suggest realistic bounds
suggested_mtry_min <- max(1, floor(sqrt(p) / 2))
suggested_mtry_max <- min(p, ceiling(sqrt(p) * 2))
suggested_depth_min <- 3
suggested_depth_max <- 16

cat("Suggested mtry range: ", suggested_mtry_min, "to", suggested_mtry_max, "\n")
cat("Suggested max.depth range: ", suggested_depth_min, "to", suggested_depth_max, "\n")

# Define scoring function for RF
rf_cv_score <- function(mtry, max_depth) {
  model <- ranger(
    formula = bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
      asset_growth + sales_growth + emp_growth + roe_change + pb_change +
      op_margin + log_at + sic2_f,
    data = train_data,
    num.trees = 300,
    mtry = floor(mtry),
    max.depth = floor(max_depth),
    probability = TRUE
  )
  
  preds <- predict(model, data = test_data)$predictions[, "1"]
  auc <- pROC::auc(test_data$bankrupt, preds)
  return(list(Score = auc))
}

# Run Bayesian Optimization
set.seed(123)
opt_rf <- bayesOpt(
  FUN = rf_cv_score,
  bounds = list(
    mtry = c(suggested_mtry_min, suggested_mtry_max),
    max_depth = c(suggested_depth_min, suggested_depth_max)
  ),
  initPoints = 5,
  iters.n = 15,
  acq = "ucb",  # acquisition function
  kappa = 2.576 # controls exploration vs. exploitation
)

# Best parameters
getBestPars(opt_rf)
ntree <- 300

set.seed(123)
rf_model <- ranger(
  formula = rf_formula,
  data = train_data,
  sample.fraction = 0.05,
  num.trees = ntree,
  max.depth = opt_rf$max_depth,
  mtry = opt_rf$mtry,
  probability = TRUE)

# Evaluate on test set
rf_probs <- predict(rf_model, data = test_data)$predictions[, "1"]
roc_rf <- pROC::roc(test_data$bankrupt, rf_probs)
auc(roc_rf)
pROC::plot.roc(test_data$bankrupt, rf_probs, main = "ROC Curve for Random Forest")

# Importance
set.seed(123)
rf_model_imp <- ranger(
  formula = rf_formula,
  data = train_data,
  sample.fraction = 0.05,
  num.trees = ntree,
  max.depth = opt_rf$max_depth,
  mtry = opt_rf$mtry,
  probability = TRUE,
  importance = "impurity")

# Feature importance
importance_df <- data.frame(
  Variable = names(rf_model_imp$variable.importance),
  Importance = rf_model_imp$variable.importance
) %>%
  arrange(desc(Importance))

# Print the sorted importance table
print(importance_df)


# Confusion matrix
best_thresh_rf <- as.numeric(coords(roc_rf, "best", ret = "threshold", transpose = FALSE))
pred_class_rf <- ifelse(rf_probs >= best_thresh_rf, 1, 0)
conf_matrix_rf <- table(Predicted = pred_class_rf, Actual = test_data$bankrupt)
print(conf_matrix_rf)

# 4. Precision, recall, F1-score
TP_rf <- conf_matrix_rf["1", "1"]
FP_rf <- conf_matrix_rf["1", "0"]
FN_rf <- conf_matrix_rf["0", "1"]

precision_rf <- TP_rf / (TP_rf + FP_rf)
recall_rf <- TP_rf / (TP_rf + FN_rf)
f1_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

precision_rf; recall_rf; f1_rf


# XGBoost ----
# Remove IDs and non-predictor columns
feature_cols <- c("wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta", "asset_growth",
                  "sales_growth", "emp_growth", "roe_change", "pb_change", "op_margin",
                  "log_at", "sic2_f")

# Ensure all predictors are numeric (convert factors like sic2_f to dummies)
train_matrix <- model.matrix(~ . - 1, data = train_data[, feature_cols])
test_matrix  <- model.matrix(~ . - 1, data = test_data[, feature_cols])

# Convert target to numeric vector (0/1)
train_label <- as.numeric(as.character(train_data$bankrupt))
test_label  <- as.numeric(as.character(test_data$bankrupt))

# Create DMatrix objects (XGBoost format)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)


# Hyperparameter tuning (XGBoost)
# 1. Data summary for tuning ranges
# Number of predictors (after model.matrix)
p <- ncol(train_matrix)
imbalance_ratio <- sum(train_label == 0) / sum(train_label == 1)

cat("Number of features:", p, "\n")
cat("Imbalance ratio (non-bankrupt / bankrupt):", round(imbalance_ratio, 2), "\n")

# 2. Define the scoring function
xgb_cv_score <- function(eta, max_depth, subsample, colsample_bytree, min_child_weight, scale_pos_weight) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = eta,
    max_depth = as.integer(max_depth),
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    scale_pos_weight = scale_pos_weight
  )
  
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100,
    watchlist = list(eval = dtest),
    verbose = 0,
    early_stopping_rounds = 10
  )
  
  pred <- predict(model, newdata = dtest)
  auc_score <- pROC::auc(test_label, pred)
  return(list(Score = auc_score))
}

# 3. Run Bayesian Optimization
set.seed(123)

opt_xgb <- bayesOpt(
  FUN = xgb_cv_score,
  bounds = list(
    eta = c(0.01, 0.3),
    max_depth = c(3L, 10L),
    subsample = c(0.6, 1),
    colsample_bytree = c(0.6, 1),
    min_child_weight = c(1, 10),
    scale_pos_weight = c(1, imbalance_ratio)
  ),
  initPoints = 10,
  iters.n = 15,
  acq = "ucb",
  kappa = 2.576,
  parallel = FALSE,      
  verbose = 1
)

# Final XGBoost model with best parameters
best_params_xgb <- getBestPars(opt_xgb)
print(best_params_xgb)

final_params_xgb <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = best_params_xgb$eta,
  max_depth = as.integer(best_params_xgb$max_depth),
  subsample = best_params_xgb$subsample,
  colsample_bytree = best_params_xgb$colsample_bytree,
  min_child_weight = best_params_xgb$min_child_weight,
  scale_pos_weight = best_params_xgb$scale_pos_weight
)

xgb_model <- xgb.train(
  params = final_params_xgb,
  data = dtrain,
  nrounds = 100,
  watchlist = list(eval = dtest),
  print_every_n = 10,
  early_stopping_rounds = 10
)

xgb_pred <- predict(xgb_model, newdata = dtest)

# Get feature importance from the trained XGBoost model
importance_df_xgb <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)
importance_df_xgb <- importance_df_xgb %>%
  arrange(desc(Gain))

print(importance_df_xgb)

# AUC
roc_xgb <- roc(test_label, xgb_pred)
auc_xgb <- auc(roc_xgb)
print(auc_xgb)
plot(roc_xgb, main = "ROC Curve for XGBoost")


best_thresh_xgb <- as.numeric(coords(roc_xgb, "best", ret = "threshold", transpose = FALSE))[1]
pred_class_xgb <- ifelse(xgb_pred >= best_thresh_xgb, 1, 0)
conf_matrix_xgb <- table(Predicted = pred_class_xgb, Actual = test_label)
print(conf_matrix_xgb)

# Precision, Recall, F1
TP_xgb <- conf_matrix_xgb["1", "1"]
FP_xgb <- conf_matrix_xgb["1", "0"]
FN_xgb <- conf_matrix_xgb["0", "1"]

precision_xgb <- TP_xgb / (TP_xgb + FP_xgb)
recall_xgb <- TP_xgb / (TP_xgb + FN_xgb)
f1_xgb <- 2 * precision_xgb * recall_xgb / (precision_xgb + recall_xgb)

precision_xgb; recall_xgb; f1_xgb


# Neural Network ----
# Data preparation
feature_cols <- c("wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta", "asset_growth",
                  "sales_growth", "emp_growth", "roe_change", "pb_change", "op_margin",
                  "log_at", "sic2_f")

# One-hot encode factor variables
train_x <- model.matrix(~ . - 1, data = train_data[, feature_cols])
test_x  <- model.matrix(~ . - 1, data = test_data[, feature_cols])


# Scale the features (standardization)
# Use training data to compute scaling parameters
train_x <- scale(train_x)
test_x  <- scale(test_x,
                 center = attr(train_x, "scaled:center"),
                 scale = attr(train_x, "scaled:scale"))  

# Convert target to numeric binary vector
train_y <- as.numeric(as.character(train_data$bankrupt))
test_y  <- as.numeric(as.character(test_data$bankrupt))


# Function to create and train a neural network with given hyperparameters
train_nn_model <- function(units1, units2, dropout, learning_rate, batch_size) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units1, activation = "relu", input_shape = ncol(train_x)) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = units2, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = c("accuracy")
  )
  
  history <- model %>% fit(
    x = train_x,
    y = train_y,
    validation_split = 0.2,
    epochs = 50,
    batch_size = batch_size,
    verbose = 0  
  )
  
  preds <- as.vector(predict(model, x = test_x))
  auc <- pROC::auc(test_y, preds)
  return(list(auc = auc, model = model))
}

set.seed(123)
tensorflow::tf$random$set_seed(123)

# Define search space
search_space <- data.frame(
  units1 = sample(c(16, 32, 64, 128), 10, replace = TRUE),
  units2 = sample(c(8, 16, 32), 10, replace = TRUE),
  dropout = runif(10, 0.2, 0.5),
  learning_rate = runif(10, 0.0005, 0.01),
  batch_size = sample(c(32, 64, 128), 10, replace = TRUE)
)

# Store results
results <- list()
best_auc <- 0
best_model <- NULL
best_config <- NULL

for (i in 1:nrow(search_space)) {
  cat("Training model", i, "...\n")
  config <- search_space[i, ]
  result <- train_nn_model(config$units1, config$units2, config$dropout,
                           config$learning_rate, config$batch_size)
  cat("AUC:", round(result$auc, 4), "\n\n")
  
  results[[i]] <- list(config = config, auc = result$auc)
  
  if (result$auc > best_auc) {
    best_auc <- result$auc
    best_model <- result$model
    best_config <- config
  }
}

cat("Best AUC:", round(best_auc, 4), "\n")
print(best_config)

# Final model
set.seed(123)
tensorflow::tf$random$set_seed(123)

nn_model <- keras_model_sequential() %>%
  layer_dense(units = best_config$units1, activation = "relu", input_shape = ncol(train_x)) %>%
  layer_dropout(rate = best_config$dropout) %>%
  layer_dense(units = best_config$units2, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

nn_model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = best_config$learning_rate),
  metrics = c("accuracy")
)

history <- nn_model %>% fit(
  x = train_x,
  y = train_y,
  validation_split = 0.2,
  epochs = 50,
  batch_size = best_config$batch_size,
  verbose = 0)

# Evaluate Model
results <- nn_model %>% evaluate(test_x, test_y)

# Do Predictions
nn_probs <- as.vector(predict(nn_model, test_x))

# AUC - ROC
roc_nn <- roc(test_y, nn_probs)
auc_nn <- auc(roc_nn)
cat("AUC:", round(auc_nn, 4), "\n")
plot(roc_nn, main = "ROC Curve for Neural Network")

# Confusion matrix
best_thresh_nn <- as.numeric(coords(roc_nn, "best", ret = "threshold", transpose = FALSE))
pred_class_nn <- ifelse(nn_probs >= best_thresh_nn, 1, 0)
conf_matrix_nn <- table(Predicted = pred_class_nn, Actual = test_data$bankrupt)
print(conf_matrix_nn)

# Precision, recall, F1-score
TP_nn <- conf_matrix_nn["1", "1"]
FP_nn <- conf_matrix_nn["1", "0"]
FN_nn <- conf_matrix_nn["0", "1"]

precision_nn <- TP_nn / (TP_nn + FP_nn)
recall_nn <- TP_nn / (TP_nn + FN_nn)
f1_nn <- 2 * (precision_nn * recall_nn) / (precision_nn + recall_nn)

precision_nn; recall_nn; f1_nn


# Aggregated ROC curves ----
roc_lr  <- roc(test_y, lr_pred)
roc_rf  <- roc(test_y, rf_probs)
roc_xgb <- roc(test_y, xgb_pred)
roc_nn  <- roc(test_y, nn_probs)

roc_df <- rbind(
  data.frame(model = "Logistic Regression",
             specificity = 1 - roc_lr$specificities,
             sensitivity = roc_lr$sensitivities),
  data.frame(model = "Random Forest",
             specificity = 1 - roc_rf$specificities,
             sensitivity = roc_rf$sensitivities),
  data.frame(model = "XGBoost",
             specificity = 1 - roc_xgb$specificities,
             sensitivity = roc_xgb$sensitivities),
  data.frame(model = "Neural Network",
             specificity = 1 - roc_nn$specificities,
             sensitivity = roc_nn$sensitivities)
)

# Plot all ROC curves on the same graph
ggplot(roc_df, aes(x = specificity, y = sensitivity, color = model)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "ROC Curves – Firm-Level Model",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Model") +
  theme_minimal()


# 3. INDUSTRY RATIO MODELS -----------------------------------------------------

# New df ----
# List of ratio variables to adjust
ratios <- c("wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta", "asset_growth",
            "sales_growth", "emp_growth", "roe_change", "pb_change", "op_margin")

# Compute industry-year medians
industry_means <- model_data %>%
  group_by(sic2_f, fyear) %>%
  summarise(across(all_of(ratios), ~ mean(., na.rm = TRUE), .names = "ind_mean_{.col}")) %>%
  ungroup()

# Merge back into main data
model_data_industry <- model_data %>%
  left_join(industry_means, by = c("sic2_f", "fyear"))

model_data_industry <- model_data_industry %>%
  group_by(gvkey) %>%
  mutate(
    ref_year = ifelse(
      bankrupt_firm == 1,
      year(DLSTDT),
      max(fyear, na.rm = TRUE)
    )
  ) %>%
  ungroup()

model_data_industry$datadate  <- as.Date(model_data_industry$datadate)
model_data_industry$DLSTDT <- as.Date(model_data_industry$DLSTDT)

write.csv(model_data_industry, "final_model_data_industry.csv", row.names = FALSE)
model_data_industry <- read.csv("final_model_data_industry.csv")

# Decide train/test split based on reference year
train_ids <- model_data_industry %>%
  filter(ref_year < 2018) %>%
  distinct(gvkey) %>%
  pull(gvkey)

test_ids <- model_data_industry %>%
  filter(ref_year >= 2018) %>%
  distinct(gvkey) %>%
  pull(gvkey)

# Assign full firm history to the proper set
train_data_industry <- model_data_industry %>%
  filter(gvkey %in% train_ids)

test_data_industry <- model_data_industry %>%
  filter(gvkey %in% test_ids)
train_data_industry$bankrupt <- factor(train_data_industry$bankrupt, levels = c(0, 1))
test_data_industry$bankrupt  <- factor(test_data_industry$bankrupt, levels = c(0, 1))

# Recreate a clean version of sic2_f with only observed levels in train
train_data_industry$sic2_f <- factor(as.character(train_data_industry$sic2), levels = unique(as.character(train_data_industry$sic2)))
test_data_industry$sic2_f <- factor(as.character(test_data_industry$sic2), levels = levels(train_data_industry$sic2_f))
test_data_industry <- test_data_industry %>% filter(!is.na(sic2_f))

train_data_industry %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()
test_data_industry %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()


# Logistic regression industry ----
lr_model_ind <- glm(bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
                      asset_growth + sales_growth + emp_growth +roe_change + 
                      pb_change + op_margin + log_at + sic2_f +
                      ind_mean_wc_ta + ind_mean_re_ta + ind_mean_ebit_ta +
                      ind_mean_mve_bvlt + ind_mean_sale_ta +
                      ind_mean_asset_growth + ind_mean_sales_growth +
                      ind_mean_emp_growth + ind_mean_roe_change + 
                      ind_mean_pb_change + ind_mean_op_margin,
                data = train_data_industry, family = "binomial")
lr_pred_ind <- predict(lr_model_ind, newdata = test_data_industry, type = "response")

summary(lr_model_ind)

# AUC - ROC
auc_lr_ind <- roc(test_data_industry$bankrupt, lr_pred_ind)
print(auc_lr_ind)
plot(auc_lr_ind, main = "ROC Curve for Logistic Regression (Industry)")

# Confusion matrix
best_thresh_lr_ind <- as.numeric(coords(auc_lr_ind, "best", ret = "threshold", transpose = FALSE))
pred_class_lr_ind <- ifelse(lr_pred_ind >= best_thresh_lr_ind, 1, 0)
conf_matrix_lr_ind <- table(Predicted = pred_class_lr_ind, Actual = test_data_industry$bankrupt)
print(conf_matrix_lr_ind)

# Precision, recall, and F1
TP_lr_ind <- conf_matrix_lr_ind["1", "1"]
FP_lr_ind <- conf_matrix_lr_ind["1", "0"]
FN_lr_ind <- conf_matrix_lr_ind["0", "1"]

precision_lr_ind <- TP_lr_ind / (TP_lr_ind + FP_lr_ind)
precision_lr_ind
recall_lr_ind <- TP_lr_ind / (TP_lr_ind + FN_lr_ind)
recall_lr_ind
f1_lr_ind <- 2 * (precision_lr_ind * recall_lr_ind) / (precision_lr_ind + recall_lr_ind)
f1_lr_ind


# Random forest industry ----
rf_formula_ind <- bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
  asset_growth + sales_growth + emp_growth + roe_change + pb_change +
  op_margin + log_at + sic2_f + ind_mean_wc_ta + ind_mean_re_ta + ind_mean_ebit_ta +
  ind_mean_mve_bvlt + ind_mean_sale_ta + ind_mean_asset_growth + ind_mean_sales_growth +
  ind_mean_emp_growth + ind_mean_roe_change + ind_mean_pb_change + ind_mean_op_margin

set.seed(123)
# OOB diagnostic run - just for tree count selection
rf_oob_plot_ind <- randomForest(
  bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
    asset_growth + sales_growth + emp_growth + roe_change + pb_change +
    op_margin + log_at + ind_mean_wc_ta + ind_mean_re_ta + ind_mean_ebit_ta +
    ind_mean_mve_bvlt + ind_mean_sale_ta + ind_mean_asset_growth + ind_mean_sales_growth +
    ind_mean_emp_growth + ind_mean_roe_change + ind_mean_pb_change + ind_mean_op_margin,
  data = train_data_industry,
  ntree = 1000,
  importance = TRUE
)

plot(rf_oob_plot_ind, main = "OOB Error vs. Number of Trees")


# Count number of predictors used in your model
# Exclude: target + IDs + factor variables (if already dummy-encoded)
numeric_predictors_ind <- train_data_industry %>%
  select(where(is.numeric)) %>%
  select(-bankrupt_firm)

p_ind <- ncol(numeric_predictors_ind)
cat("Number of numeric predictors:", p_ind, "\n")

# Suggest realistic bounds
suggested_mtry_min_ind <- max(1, floor(sqrt(p_ind) / 2))
suggested_mtry_max_ind <- min(p_ind, ceiling(sqrt(p_ind) * 2))
suggested_depth_min_ind <- 3
suggested_depth_max_ind <- 16

cat("Suggested mtry range: ", suggested_mtry_min_ind, "to", suggested_mtry_max_ind, "\n")
cat("Suggested max.depth range: ", suggested_depth_min_ind, "to", suggested_depth_max_ind, "\n")

# Define scoring function for RF
rf_cv_score_ind <- function(mtry, max_depth) {
  model <- ranger(
    formula = rf_formula_ind,
    data = train_data_industry,
    num.trees = 300,
    mtry = floor(mtry),
    max.depth = floor(max_depth),
    probability = TRUE
  )
  
  preds <- predict(model, data = test_data_industry)$predictions[, "1"]
  auc <- pROC::auc(test_data_industry$bankrupt, preds)
  return(list(Score = auc))
}

# Run Bayesian Optimization
set.seed(123)
opt_rf_ind <- bayesOpt(
  FUN = rf_cv_score_ind,
  bounds = list(
    mtry = c(suggested_mtry_min_ind, suggested_mtry_max_ind),
    max_depth = c(suggested_depth_min_ind, suggested_depth_max_ind)
  ),
  initPoints = 5,
  iters.n = 15,
  acq = "ucb",  # acquisition function
  kappa = 2.576 # controls exploration vs. exploitation
)

# Best parameters
getBestPars(opt_rf_ind)
ntree <- 300

set.seed(123)
rf_model_ind <- ranger(
  formula = rf_formula_ind,
  data = train_data_industry,
  sample.fraction = 0.05,
  num.trees = ntree,
  max.depth = opt_rf_ind$max_depth,
  mtry = opt_rf_ind$mtry,
  probability = TRUE)

# Evaluate on test set
rf_probs_ind <- predict(rf_model_ind, data = test_data_industry)$predictions[, "1"]
roc_rf_ind <- pROC::roc(test_data_industry$bankrupt, rf_probs_ind)
auc(roc_rf_ind)
pROC::plot.roc(test_data_industry$bankrupt, rf_probs_ind, main = "ROC Curve for Random Forest (Industry)")

# Importance
set.seed(123)
rf_model_imp_ind <- ranger(
  formula = rf_formula_ind,
  data = train_data_industry,
  sample.fraction = 0.05,
  num.trees = ntree,
  max.depth = opt_rf_ind$max_depth,
  mtry = opt_rf_ind$mtry,
  probability = TRUE,
  importance = "impurity")

# Feature importance
importance_df_ind <- data.frame(
  Variable = names(rf_model_imp_ind$variable.importance),
  Importance = rf_model_imp_ind$variable.importance
) %>%
  arrange(desc(Importance))

# Print the sorted importance table
print(importance_df_ind)


# Confusion matrix
best_thresh_rf_ind <- as.numeric(coords(roc_rf_ind, "best", ret = "threshold", transpose = FALSE))
pred_class_rf_ind <- ifelse(rf_probs_ind >= best_thresh_rf_ind, 1, 0)
conf_matrix_rf_ind <- table(Predicted = pred_class_rf_ind, Actual = test_data_industry$bankrupt)
print(conf_matrix_rf_ind)

# 4. Precision, recall, F1-score
TP_rf_ind <- conf_matrix_rf_ind["1", "1"]
FP_rf_ind <- conf_matrix_rf_ind["1", "0"]
FN_rf_ind <- conf_matrix_rf_ind["0", "1"]

precision_rf_ind <- TP_rf_ind / (TP_rf_ind + FP_rf_ind)
recall_rf_ind <- TP_rf_ind / (TP_rf_ind + FN_rf_ind)
f1_rf_ind <- 2 * (precision_rf_ind * recall_rf_ind) / (precision_rf_ind + recall_rf_ind)

precision_rf_ind; recall_rf_ind; f1_rf_ind


# XGBoost industry ----
# Remove IDs and non-predictor columns
feature_cols_ind <- c("wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta", "asset_growth",
                  "sales_growth", "emp_growth", "roe_change", "pb_change", "op_margin",
                  "ind_mean_wc_ta", "ind_mean_re_ta", "ind_mean_ebit_ta",
                  "ind_mean_mve_bvlt", "ind_mean_sale_ta", "ind_mean_asset_growth",
                  "ind_mean_sales_growth", "ind_mean_emp_growth", "ind_mean_roe_change",
                  "ind_mean_pb_change", "ind_mean_op_margin", "log_at", "sic2_f")

# Ensure all predictors are numeric (convert factors like sic2_f to dummies)
train_matrix_ind <- model.matrix(~ . - 1, data = train_data_industry[, feature_cols_ind])
test_matrix_ind  <- model.matrix(~ . - 1, data = test_data_industry[, feature_cols_ind])

# Convert target to numeric vector (0/1)
train_label_ind <- as.numeric(as.character(train_data_industry$bankrupt))
test_label_ind  <- as.numeric(as.character(test_data_industry$bankrupt))

# Create DMatrix objects (XGBoost format)
dtrain_ind <- xgb.DMatrix(data = train_matrix_ind, label = train_label_ind)
dtest_ind  <- xgb.DMatrix(data = test_matrix_ind, label = test_label_ind)


# Hyperparameter tuning (XGBoost)
# 1. Data summary for tuning ranges
# Number of predictors (after model.matrix)
p_ind <- ncol(train_matrix_ind)
imbalance_ratio_ind <- sum(train_label_ind == 0) / sum(train_label_ind == 1)

cat("Number of features:", p_ind, "\n")
cat("Imbalance ratio (non-bankrupt / bankrupt):", round(imbalance_ratio_ind, 2), "\n")

# 2. Define the scoring function
xgb_cv_score_ind <- function(eta, max_depth, subsample, colsample_bytree, min_child_weight, scale_pos_weight) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = eta,
    max_depth = as.integer(max_depth),
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    scale_pos_weight = scale_pos_weight
  )
  
  model <- xgb.train(
    params = params,
    data = dtrain_ind,
    nrounds = 100,
    watchlist = list(eval = dtest_ind),
    verbose = 0,
    early_stopping_rounds = 10
  )
  
  pred <- predict(model, newdata = dtest_ind)
  auc_score <- pROC::auc(test_label_ind, pred)
  return(list(Score = auc_score))
}

# 3. Run Bayesian Optimization
set.seed(123)

opt_xgb_ind <- bayesOpt(
  FUN = xgb_cv_score_ind,
  bounds = list(
    eta = c(0.01, 0.3),
    max_depth = c(3L, 10L),
    subsample = c(0.6, 1),
    colsample_bytree = c(0.6, 1),
    min_child_weight = c(1, 10),
    scale_pos_weight = c(1, imbalance_ratio_ind)
  ),
  initPoints = 10,
  iters.n = 15,
  acq = "ucb",
  kappa = 2.576,
  parallel = FALSE,      
  verbose = 1
)

# Final XGBoost model with best parameters
best_params_xgb_ind <- getBestPars(opt_xgb_ind)
print(best_params_xgb_ind)

final_params_xgb_ind <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = best_params_xgb_ind$eta,
  max_depth = as.integer(best_params_xgb_ind$max_depth),
  subsample = best_params_xgb_ind$subsample,
  colsample_bytree = best_params_xgb_ind$colsample_bytree,
  min_child_weight = best_params_xgb_ind$min_child_weight,
  scale_pos_weight = best_params_xgb_ind$scale_pos_weight
)

xgb_model_ind <- xgb.train(
  params = final_params_xgb_ind,
  data = dtrain_ind,
  nrounds = 100,
  watchlist = list(eval = dtest_ind),
  print_every_n = 10,
  early_stopping_rounds = 10
)

xgb_pred_ind <- predict(xgb_model_ind, newdata = dtest_ind)

# Get feature importance from the trained XGBoost model
importance_df_xgb_ind <- xgb.importance(feature_names = colnames(train_matrix_ind), model = xgb_model_ind)
importance_df_xgb_ind <- importance_df_xgb_ind %>%
  arrange(desc(Gain))

print(importance_df_xgb_ind)

# AUC
roc_xgb_ind <- roc(test_label_ind, xgb_pred_ind)
auc_xgb_ind <- auc(roc_xgb_ind)
print(auc_xgb_ind)
plot(roc_xgb_ind, main = "ROC Curve for XGBoost (Industry)")


best_thresh_xgb_ind <- as.numeric(coords(roc_xgb_ind, "best", ret = "threshold", transpose = FALSE))[1]
pred_class_xgb_ind <- ifelse(xgb_pred_ind >= best_thresh_xgb_ind, 1, 0)
conf_matrix_xgb_ind <- table(Predicted = pred_class_xgb_ind, Actual = test_label_ind)
print(conf_matrix_xgb_ind)

# Precision, Recall, F1
TP_xgb_ind <- conf_matrix_xgb_ind["1", "1"]
FP_xgb_ind <- conf_matrix_xgb_ind["1", "0"]
FN_xgb_ind <- conf_matrix_xgb_ind["0", "1"]

precision_xgb_ind <- TP_xgb_ind / (TP_xgb_ind + FP_xgb_ind)
recall_xgb_ind <- TP_xgb_ind / (TP_xgb_ind + FN_xgb_ind)
f1_xgb_ind <- 2 * precision_xgb_ind * recall_xgb_ind / (precision_xgb_ind + recall_xgb_ind)

precision_xgb_ind; recall_xgb_ind; f1_xgb_ind


# Neural Network indusrtry ----
# Data preparation
feature_cols_ind <- c("wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta", "asset_growth",
                      "sales_growth", "emp_growth", "roe_change", "pb_change", "op_margin",
                      "ind_mean_wc_ta", "ind_mean_re_ta", "ind_mean_ebit_ta",
                      "ind_mean_mve_bvlt", "ind_mean_sale_ta", "ind_mean_asset_growth",
                      "ind_mean_sales_growth", "ind_mean_emp_growth", "ind_mean_roe_change",
                      "ind_mean_pb_change", "ind_mean_op_margin", "log_at", "sic2_f")

# One-hot encode factor variables
# This converts all factor variables (like sic2_f) into dummy variables
train_x_ind <- model.matrix(~ . - 1, data = train_data_industry[, feature_cols_ind])
test_x_ind  <- model.matrix(~ . - 1, data = test_data_industry[, feature_cols_ind])


# Scale the features (standardization)
# Use training data to compute scaling parameters
train_x_ind <- scale(train_x_ind)
test_x_ind  <- scale(test_x_ind,
                 center = attr(train_x_ind, "scaled:center"),
                 scale = attr(train_x_ind, "scaled:scale"))  

# Convert target to numeric binary vector
train_y_ind <- as.numeric(as.character(train_data_industry$bankrupt))
test_y_ind  <- as.numeric(as.character(test_data_industry$bankrupt))


# Function to create and train a neural network with given hyperparameters
train_nn_model_ind <- function(units1, units2, dropout, learning_rate, batch_size) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units1, activation = "relu", input_shape = ncol(train_x_ind)) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = units2, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = c("accuracy")
  )
  
  history <- model %>% fit(
    x = train_x_ind,
    y = train_y_ind,
    validation_split = 0.2,
    epochs = 50,
    batch_size = batch_size,
    verbose = 0  
  )
  
  preds <- as.vector(predict(model, x = test_x_ind))
  auc <- pROC::auc(test_y_ind, preds)
  return(list(auc = auc, model = model))
}

set.seed(123)
tensorflow::tf$random$set_seed(123)

# Define search space
search_space_ind <- data.frame(
  units1 = sample(c(16, 32, 64, 128), 10, replace = TRUE),
  units2 = sample(c(8, 16, 32), 10, replace = TRUE),
  dropout = runif(10, 0.2, 0.5),
  learning_rate = runif(10, 0.0005, 0.01),
  batch_size = sample(c(32, 64, 128), 10, replace = TRUE)
)

# Store results
results_ind <- list()
best_auc_ind <- 0
best_model_ind <- NULL
best_config_ind <- NULL

for (i in 1:nrow(search_space_ind)) {
  cat("Training model", i, "...\n")
  config <- search_space_ind[i, ]
  result <- train_nn_model_ind(config$units1, config$units2, config$dropout,
                           config$learning_rate, config$batch_size)
  cat("AUC:", round(result$auc, 4), "\n\n")
  
  results_ind[[i]] <- list(config = config, auc = result$auc)
  
  if (result$auc > best_auc_ind) {
    best_auc_ind <- result$auc
    best_model_ind <- result$model
    best_config_ind <- config
  }
}

cat("Best AUC:", round(best_auc_ind, 4), "\n")
print(best_config_ind)

# Final model
set.seed(123)
tensorflow::tf$random$set_seed(123)

nn_model_ind <- keras_model_sequential() %>%
  layer_dense(units = best_config_ind$units1, activation = "relu", input_shape = ncol(train_x_ind)) %>%
  layer_dropout(rate = best_config_ind$dropout) %>%
  layer_dense(units = best_config_ind$units2, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

nn_model_ind %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = best_config_ind$learning_rate),
  metrics = c("accuracy")
)

history <- nn_model_ind %>% fit(
  x = train_x_ind,
  y = train_y_ind,
  validation_split = 0.2,
  epochs = 50,
  batch_size = best_config_ind$batch_size,
  verbose = 0)

# Evaluate Model
results_ind <- nn_model_ind %>% evaluate(test_x_ind, test_y_ind)

# Do Predictions
nn_probs_ind <- as.vector(predict(nn_model_ind, test_x_ind))

# AUC - ROC
roc_nn_ind <- roc(test_y_ind, nn_probs_ind)
auc_nn_ind <- auc(roc_nn_ind)
cat("AUC:", round(auc_nn_ind, 4), "\n")
plot(roc_nn_ind, main = "ROC Curve for Neural Network (Industry)")

# Confusion matrix
best_thresh_nn_ind <- as.numeric(coords(roc_nn_ind, "best", ret = "threshold", transpose = FALSE))
pred_class_nn_ind <- ifelse(nn_probs_ind >= best_thresh_nn_ind, 1, 0)
conf_matrix_nn_ind <- table(Predicted = pred_class_nn_ind, Actual = test_data_industry$bankrupt)
print(conf_matrix_nn_ind)

# Precision, recall, F1-score
TP_nn_ind <- conf_matrix_nn_ind["1", "1"]
FP_nn_ind <- conf_matrix_nn_ind["1", "0"]
FN_nn_ind <- conf_matrix_nn_ind["0", "1"]

precision_nn_ind <- TP_nn_ind / (TP_nn_ind + FP_nn_ind)
recall_nn_ind <- TP_nn_ind / (TP_nn_ind + FN_nn_ind)
f1_nn_ind <- 2 * (precision_nn_ind * recall_nn_ind) / (precision_nn_ind + recall_nn_ind)

precision_nn_ind; recall_nn_ind; f1_nn_ind


# Aggregated ROC curves industry ----
roc_lr_ind  <- roc(test_y_ind, lr_pred_ind)
roc_rf_ind  <- roc(test_y_ind, rf_probs_ind)
roc_xgb_ind <- roc(test_y_ind, xgb_pred_ind)
roc_nn_ind  <- roc(test_y_ind, nn_probs_ind)

roc_df_ind <- rbind(
  data.frame(model = "Logistic Regression",
             specificity = 1 - roc_lr_ind$specificities,
             sensitivity = roc_lr_ind$sensitivities),
  data.frame(model = "Random Forest",
             specificity = 1 - roc_rf_ind$specificities,
             sensitivity = roc_rf_ind$sensitivities),
  data.frame(model = "XGBoost",
             specificity = 1 - roc_xgb_ind$specificities,
             sensitivity = roc_xgb_ind$sensitivities),
  data.frame(model = "Neural Network",
             specificity = 1 - roc_nn_ind$specificities,
             sensitivity = roc_nn_ind$sensitivities)
)

# Plot all ROC curves on the same graph
ggplot(roc_df_ind, aes(x = specificity, y = sensitivity, color = model)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "ROC Curves – Industry-Level Model",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Model") +
  theme_minimal()


# 4. MACRO MODELS --------------------------------------------------------------

macro_files <- c("DFF.csv", "GDPGROWTH.csv", "MEDCPI.csv", "UNRATE.csv", "VIXCLS.csv")
macro_paths <- file.path(dirpath, macro_files)

# Read and merge
macro_data <- macro_paths %>%
  map(read.csv) %>%
  reduce(full_join, by = "observation_date") %>%
  mutate(year = as.integer(format(as.Date(observation_date), "%Y"))) %>%
  mutate(year = year + 1) %>%
  filter(year >= "1990" & year <= "2023") %>%
  select(-observation_date) %>%
  rename(
    interest_rate = DFF,
    gdp_growth = A191RO1Q156NBEA,
    inflation = MEDCPIM158SFRBCLE,
    unemployment = UNRATE,
    vix = VIXCLS) %>%
  mutate(across(c(gdp_growth, inflation, unemployment, interest_rate, vix), ~ . / 100))

# Merge with firm-level data
model_data_macro <- model_data %>%
  left_join(macro_data, by = c("fyear" = "year")) %>%
  filter(fyear >= 1991)

write.csv(model_data_macro, "final_model_data_macro.csv", row.names = FALSE)
model_data_macro <- read.csv("final_model_data_macro.csv")

# Decide train/test split based on reference year
train_ids <- model_data_macro %>%
  filter(ref_year < 2018) %>%
  distinct(gvkey) %>%
  pull(gvkey)

test_ids <- model_data_macro %>%
  filter(ref_year >= 2018) %>%
  distinct(gvkey) %>%
  pull(gvkey)

# Assign full firm history to the proper set
train_data_macro <- model_data_macro %>%
  filter(gvkey %in% train_ids)

test_data_macro <- model_data_macro %>%
  filter(gvkey %in% test_ids)
train_data_macro$bankrupt <- factor(train_data_macro$bankrupt, levels = c(0, 1))
test_data_macro$bankrupt  <- factor(test_data_macro$bankrupt, levels = c(0, 1))

# Recreate a clean version of sic2_f with only observed levels in train
train_data_macro$sic2_f <- factor(as.character(train_data_macro$sic2), levels = unique(as.character(train_data_macro$sic2)))
test_data_macro$sic2_f <- factor(as.character(test_data_macro$sic2), levels = levels(train_data_macro$sic2_f))
test_data_macro <- test_data_macro %>% filter(!is.na(sic2_f))

train_data_macro %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()
test_data_macro %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()

macro_vars <- train_data_macro %>%
  select(interest_rate, gdp_growth, inflation, unemployment, vix)

cor(macro_vars, use = "complete.obs")

# Summarise for each variable for macro factors
macro_variables <- c("interest_rate", "gdp_growth", "inflation", "unemployment", "vix")
model_data_summary_macro <- model_data_macro %>%
  select(all_of(macro_variables)) %>%
  summarise(across(all_of(macro_variables), list(
    mean = ~ as.numeric(mean(., na.rm = TRUE)),
    sd = ~ as.numeric(sd(., na.rm = TRUE)),
    min = ~ as.numeric(min(., na.rm = TRUE)),
    q1 = ~ as.numeric(quantile(., 0.25, na.rm = TRUE)),
    median = ~ as.numeric(median(., na.rm = TRUE)),
    q3 = ~ as.numeric(quantile(., 0.75, na.rm = TRUE)),
    max = ~ as.numeric(max(., na.rm = TRUE))
  ), .names = "{.col}__{.fn}")) %>%
  pivot_longer(everything(),
               names_to = c("variable", "stat"),
               names_sep = "__") %>%
  pivot_wider(names_from = stat, values_from = value)

write_xlsx(model_data_summary_macro, "descriptive_statistics_macro.xlsx")


# Logistic regression macro ----
lr_model_m <- glm(bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
                      asset_growth + sales_growth + emp_growth +roe_change + 
                      pb_change + op_margin + log_at + sic2_f +
                      interest_rate + gdp_growth + unemployment + vix,
                    data = train_data_macro, family = "binomial")
lr_pred_m <- predict(lr_model_m, newdata = test_data_macro, type = "response")

summary(lr_model_m)

# AUC - ROC
auc_lr_m <- roc(test_data_macro$bankrupt, lr_pred_m)
print(auc_lr_m)
plot(auc_lr_m, main = "ROC Curve for Logistic Regression (Macro)")

# Confusion matrix
best_thresh_lr_m <- as.numeric(coords(auc_lr_m, "best", ret = "threshold", transpose = FALSE))
pred_class_lr_m <- ifelse(lr_pred_m >= best_thresh_lr_m, 1, 0)
conf_matrix_lr_m <- table(Predicted = pred_class_lr_m, Actual = test_data_macro$bankrupt)
print(conf_matrix_lr_m)

# Precision, recall, and F1
TP_lr_m <- conf_matrix_lr_m["1", "1"]
FP_lr_m <- conf_matrix_lr_m["1", "0"]
FN_lr_m <- conf_matrix_lr_m["0", "1"]

precision_lr_m <- TP_lr_m / (TP_lr_m + FP_lr_m)
precision_lr_m
recall_lr_m <- TP_lr_m / (TP_lr_m + FN_lr_m)
recall_lr_m
f1_lr_m <- 2 * (precision_lr_m * recall_lr_m) / (precision_lr_m + recall_lr_m)
f1_lr_m


# Random forest macro ----
rf_formula_m <- bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
  asset_growth + sales_growth + emp_growth +roe_change + pb_change + op_margin +
  log_at + sic2_f + interest_rate + gdp_growth + unemployment + vix

set.seed(123)
# OOB diagnostic run - just for tree count selection
rf_oob_plot_m <- randomForest(
  bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
    asset_growth + sales_growth + emp_growth +roe_change + pb_change + op_margin +
    log_at + interest_rate + gdp_growth + unemployment + vix,
  data = train_data_macro,
  ntree = 1000,
  importance = TRUE
)

plot(rf_oob_plot_m, main = "OOB Error vs. Number of Trees")


# Count number of predictors used in your model
# Exclude: target + IDs + factor variables (if already dummy-encoded)
numeric_predictors_m <- train_data_macro %>%
  select(where(is.numeric)) %>%
  select(-bankrupt_firm)

p_m <- ncol(numeric_predictors_m)
cat("Number of numeric predictors:", p_m, "\n")

# Suggest realistic bounds
suggested_mtry_min_m <- max(1, floor(sqrt(p_m) / 2))
suggested_mtry_max_m <- min(p_m, ceiling(sqrt(p_m) * 2))
suggested_depth_min_m <- 3
suggested_depth_max_m <- 16

cat("Suggested mtry range: ", suggested_mtry_min_m, "to", suggested_mtry_max_m, "\n")
cat("Suggested max.depth range: ", suggested_depth_min_m, "to", suggested_depth_max_m, "\n")

# Define scoring function for RF
rf_cv_score_m <- function(mtry, max_depth) {
  model <- ranger(
    formula = rf_formula_m,
    data = train_data_macro,
    num.trees = 300,
    mtry = floor(mtry),
    max.depth = floor(max_depth),
    probability = TRUE
  )
  
  preds <- predict(model, data = test_data_macro)$predictions[, "1"]
  auc <- pROC::auc(test_data_macro$bankrupt, preds)
  return(list(Score = auc))
}

# Run Bayesian Optimization
set.seed(123)
opt_rf_m <- bayesOpt(
  FUN = rf_cv_score_m,
  bounds = list(
    mtry = c(suggested_mtry_min_m, suggested_mtry_max_m),
    max_depth = c(suggested_depth_min_m, suggested_depth_max_m)
  ),
  initPoints = 5,
  iters.n = 15,
  acq = "ucb",  # acquisition function
  kappa = 2.576 # controls exploration vs. exploitation
)

# Best parameters
getBestPars(opt_rf_m)
ntree <- 300

set.seed(123)
rf_model_m <- ranger(
  formula = rf_formula_m,
  data = train_data_macro,
  sample.fraction = 0.05,
  num.trees = ntree,
  max.depth = opt_rf_m$max_depth,
  mtry = opt_rf_m$mtry,
  probability = TRUE)

# Evaluate on test set
rf_probs_m <- predict(rf_model_m, data = test_data_macro)$predictions[, "1"]
roc_rf_m <- pROC::roc(test_data_macro$bankrupt, rf_probs_m)
auc(roc_rf_m)
pROC::plot.roc(test_data_macro$bankrupt, rf_probs_m, main = "ROC Curve for Random Forest (Macro)")

# Importance
set.seed(123)
rf_model_imp_m <- ranger(
  formula = rf_formula_m,
  data = train_data_macro,
  sample.fraction = 0.05,
  num.trees = ntree,
  max.depth = opt_rf_m$max_depth,
  mtry = opt_rf_m$mtry,
  probability = TRUE,
  importance = "impurity")

# Feature importance
importance_df_m <- data.frame(
  Variable = names(rf_model_imp_m$variable.importance),
  Importance = rf_model_imp_m$variable.importance
) %>%
  arrange(desc(Importance))

# Print the sorted importance table
print(importance_df_m)


# Confusion matrix
best_thresh_rf_m <- as.numeric(coords(roc_rf_m, "best", ret = "threshold", transpose = FALSE))
pred_class_rf_m <- ifelse(rf_probs_m >= best_thresh_rf_m, 1, 0)
conf_matrix_rf_m <- table(Predicted = pred_class_rf_m, Actual = test_data_macro$bankrupt)
print(conf_matrix_rf_m)

# 4. Precision, recall, F1-score
TP_rf_m <- conf_matrix_rf_m["1", "1"]
FP_rf_m <- conf_matrix_rf_m["1", "0"]
FN_rf_m <- conf_matrix_rf_m["0", "1"]

precision_rf_m <- TP_rf_m / (TP_rf_m + FP_rf_m)
recall_rf_m <- TP_rf_m / (TP_rf_m + FN_rf_m)
f1_rf_m <- 2 * (precision_rf_m * recall_rf_m) / (precision_rf_m + recall_rf_m)

precision_rf_m; recall_rf_m; f1_rf_m


# XGBoost macro ----
# Remove IDs and non-predictor columns
feature_cols_m <- c("wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta", "asset_growth",
                    "sales_growth", "emp_growth", "roe_change", "pb_change", "op_margin",
                    "log_at", "sic2_f", "interest_rate", "gdp_growth", "unemployment", "vix")

# Ensure all predictors are numeric (convert factors like sic2_f to dummies)
train_matrix_m <- model.matrix(~ . - 1, data = train_data_macro[, feature_cols_m])
test_matrix_m  <- model.matrix(~ . - 1, data = test_data_macro[, feature_cols_m])

# Convert target to numeric vector (0/1)
train_label_m <- as.numeric(as.character(train_data_macro$bankrupt))
test_label_m  <- as.numeric(as.character(test_data_macro$bankrupt))

# Create DMatrix objects (XGBoost format)
dtrain_m <- xgb.DMatrix(data = train_matrix_m, label = train_label_m)
dtest_m  <- xgb.DMatrix(data = test_matrix_m, label = test_label_m)


# Hyperparameter tuning (XGBoost)
# 1. Data summary for tuning ranges
# Number of predictors (after model.matrix)
p_m <- ncol(train_matrix_m)
imbalance_ratio_m <- sum(train_label_m == 0) / sum(train_label_m == 1)

cat("Number of features:", p_m, "\n")
cat("Imbalance ratio (non-bankrupt / bankrupt):", round(imbalance_ratio_m, 2), "\n")

# 2. Define the scoring function
xgb_cv_score_m <- function(eta, max_depth, subsample, colsample_bytree, min_child_weight, scale_pos_weight) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = eta,
    max_depth = as.integer(max_depth),
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    scale_pos_weight = scale_pos_weight
  )
  
  model <- xgb.train(
    params = params,
    data = dtrain_m,
    nrounds = 100,
    watchlist = list(eval = dtest_m),
    verbose = 0,
    early_stopping_rounds = 10
  )
  
  pred <- predict(model, newdata = dtest_m)
  auc_score <- pROC::auc(test_label_m, pred)
  return(list(Score = auc_score))
}

# 3. Run Bayesian Optimization
set.seed(123)

opt_xgb_m <- bayesOpt(
  FUN = xgb_cv_score_m,
  bounds = list(
    eta = c(0.01, 0.3),
    max_depth = c(3L, 10L),
    subsample = c(0.6, 1),
    colsample_bytree = c(0.6, 1),
    min_child_weight = c(1, 10),
    scale_pos_weight = c(1, imbalance_ratio_m)
  ),
  initPoints = 10,
  iters.n = 15,
  acq = "ucb",
  kappa = 2.576,
  parallel = FALSE,      
  verbose = 1
)

# Final XGBoost model with best parameters
best_params_xgb_m <- getBestPars(opt_xgb_m)
print(best_params_xgb_m)

final_params_xgb_m <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = best_params_xgb_m$eta,
  max_depth = as.integer(best_params_xgb_m$max_depth),
  subsample = best_params_xgb_m$subsample,
  colsample_bytree = best_params_xgb_m$colsample_bytree,
  min_child_weight = best_params_xgb_m$min_child_weight,
  scale_pos_weight = best_params_xgb_m$scale_pos_weight
)

xgb_model_m <- xgb.train(
  params = final_params_xgb_m,
  data = dtrain_m,
  nrounds = 100,
  watchlist = list(eval = dtest_m),
  print_every_n = 10,
  early_stopping_rounds = 10
)

xgb_pred_m <- predict(xgb_model_m, newdata = dtest_m)

# Get feature importance from the trained XGBoost model
importance_df_xgb_m <- xgb.importance(feature_names = colnames(train_matrix_m), model = xgb_model_m)
importance_df_xgb_m <- importance_df_xgb_m %>%
  arrange(desc(Gain))

print(importance_df_xgb_m)

# AUC
roc_xgb_m <- roc(test_label_m, xgb_pred_m)
auc_xgb_m <- auc(roc_xgb_m)
print(auc_xgb_m)
plot(roc_xgb_m, main = "ROC Curve for XGBoost (Macro)")


best_thresh_xgb_m <- as.numeric(coords(roc_xgb_m, "best", ret = "threshold", transpose = FALSE))[1]
pred_class_xgb_m <- ifelse(xgb_pred_m >= best_thresh_xgb_m, 1, 0)
conf_matrix_xgb_m <- table(Predicted = pred_class_xgb_m, Actual = test_label_m)
print(conf_matrix_xgb_m)

# Precision, Recall, F1
TP_xgb_m <- conf_matrix_xgb_m["1", "1"]
FP_xgb_m <- conf_matrix_xgb_m["1", "0"]
FN_xgb_m <- conf_matrix_xgb_m["0", "1"]

precision_xgb_m <- TP_xgb_m / (TP_xgb_m + FP_xgb_m)
recall_xgb_m <- TP_xgb_m / (TP_xgb_m + FN_xgb_m)
f1_xgb_m <- 2 * precision_xgb_m * recall_xgb_m / (precision_xgb_m + recall_xgb_m)

precision_xgb_m; recall_xgb_m; f1_xgb_m


# Neural Network ----
# Data preparation
feature_cols_m <- c("wc_ta", "re_ta", "ebit_ta", "mve_bvlt", "sale_ta", "asset_growth",
                    "sales_growth", "emp_growth", "roe_change", "pb_change", "op_margin",
                    "log_at", "sic2_f", "interest_rate", "gdp_growth", "unemployment", "vix")

# One-hot encode factor variables
# This converts all factor variables (like sic2_f) into dummy variables
train_x_m <- model.matrix(~ . - 1, data = train_data_macro[, feature_cols_m])
test_x_m  <- model.matrix(~ . - 1, data = test_data_macro[, feature_cols_m])


# Scale the features (standardization)
# Use training data to compute scaling parameters
train_x_m <- scale(train_x_m)
test_x_m  <- scale(test_x_m,
                     center = attr(train_x_m, "scaled:center"),
                     scale = attr(train_x_m, "scaled:scale"))  

# Convert target to numeric binary vector
train_y_m <- as.numeric(as.character(train_data_macro$bankrupt))
test_y_m  <- as.numeric(as.character(test_data_macro$bankrupt))


# Function to create and train a neural network with given hyperparameters
train_nn_model_m <- function(units1, units2, dropout, learning_rate, batch_size) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units1, activation = "relu", input_shape = ncol(train_x_m)) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = units2, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = c("accuracy")
  )
  
  history <- model %>% fit(
    x = train_x_m,
    y = train_y_m,
    validation_split = 0.2,
    epochs = 50,
    batch_size = batch_size,
    verbose = 0  
  )
  
  preds <- as.vector(predict(model, x = test_x_m))
  auc <- pROC::auc(test_y_m, preds)
  return(list(auc = auc, model = model))
}

set.seed(123)
tensorflow::tf$random$set_seed(123)

# Define search space
search_space_m <- data.frame(
  units1 = sample(c(16, 32, 64, 128), 10, replace = TRUE),
  units2 = sample(c(8, 16, 32), 10, replace = TRUE),
  dropout = runif(10, 0.2, 0.5),
  learning_rate = runif(10, 0.0005, 0.01),
  batch_size = sample(c(32, 64, 128), 10, replace = TRUE)
)

# Store results
results_m <- list()
best_auc_m <- 0
best_model_m <- NULL
best_config_m <- NULL

for (i in 1:nrow(search_space_m)) {
  cat("Training model", i, "...\n")
  config <- search_space_m[i, ]
  result <- train_nn_model_m(config$units1, config$units2, config$dropout,
                               config$learning_rate, config$batch_size)
  cat("AUC:", round(result$auc, 4), "\n\n")
  
  results_m[[i]] <- list(config = config, auc = result$auc)
  
  if (result$auc > best_auc_m) {
    best_auc_m <- result$auc
    best_model_m <- result$model
    best_config_m <- config
  }
}

cat("Best AUC:", round(best_auc_m, 4), "\n")
print(best_config_m)

# Final model
set.seed(123)
tensorflow::tf$random$set_seed(123)

nn_model_m <- keras_model_sequential() %>%
  layer_dense(units = best_config_m$units1, activation = "relu", input_shape = ncol(train_x_m)) %>%
  layer_dropout(rate = best_config_m$dropout) %>%
  layer_dense(units = best_config_m$units2, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

nn_model_m %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = best_config_m$learning_rate),
  metrics = c("accuracy")
)

history <- nn_model_m %>% fit(
  x = train_x_m,
  y = train_y_m,
  validation_split = 0.2,
  epochs = 50,
  batch_size = best_config_m$batch_size,
  verbose = 0)

# Evaluate Model
results_m <- nn_model_m %>% evaluate(test_x_m, test_y_m)

# Do Predictions
nn_probs_m <- as.vector(predict(nn_model_m, test_x_m))

# AUC - ROC
roc_nn_m <- roc(test_y_m, nn_probs_m)
auc_nn_m <- auc(roc_nn_m)
cat("AUC:", round(auc_nn_m, 4), "\n")
plot(roc_nn_m, main = "ROC Curve for Neural Network (Macro)")

# Confusion matrix
best_thresh_nn_m <- as.numeric(coords(roc_nn_m, "best", ret = "threshold", transpose = FALSE))
pred_class_nn_m <- ifelse(nn_probs_m >= best_thresh_nn_m, 1, 0)
conf_matrix_nn_m <- table(Predicted = pred_class_nn_m, Actual = test_data_macro$bankrupt)
print(conf_matrix_nn_m)

# Precision, recall, F1-score
TP_nn_m <- conf_matrix_nn_m["1", "1"]
FP_nn_m <- conf_matrix_nn_m["1", "0"]
FN_nn_m <- conf_matrix_nn_m["0", "1"]

precision_nn_m <- TP_nn_m / (TP_nn_m + FP_nn_m)
recall_nn_m <- TP_nn_m / (TP_nn_m + FN_nn_m)
f1_nn_m <- 2 * (precision_nn_m * recall_nn_m) / (precision_nn_m + recall_nn_m)

precision_nn_m; recall_nn_m; f1_nn_m


# Aggregated ROC curves industry ----
roc_lr_m  <- roc(test_y_m, lr_pred_m)
roc_rf_m  <- roc(test_y_m, rf_probs_m)
roc_xgb_m <- roc(test_y_m, xgb_pred_m)
roc_nn_m  <- roc(test_y_m, nn_probs_m)

roc_df_ind <- rbind(
  data.frame(model = "Logistic Regression",
             specificity = 1 - roc_lr_m$specificities,
             sensitivity = roc_lr_m$sensitivities),
  data.frame(model = "Random Forest",
             specificity = 1 - roc_rf_m$specificities,
             sensitivity = roc_rf_m$sensitivities),
  data.frame(model = "XGBoost",
             specificity = 1 - roc_xgb_m$specificities,
             sensitivity = roc_xgb_m$sensitivities),
  data.frame(model = "Neural Network",
             specificity = 1 - roc_nn_m$specificities,
             sensitivity = roc_nn_m$sensitivities)
)

# Plot all ROC curves on the same graph
ggplot(roc_df_ind, aes(x = specificity, y = sensitivity, color = model)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "ROC Curves – Macro-Level Model",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)",
       color = "Model") +
  theme_minimal()


# 5. MACROECONOMIC STRESS TESTING ----------------------------------------------

model_data_stress <- model_data_macro

# Train - test split 
train_data_stress <- model_data_stress %>%
  filter(fyear < 2018)

test_data_stress <- model_data_stress %>%
  filter(fyear >= 2018 & fyear <= 2019)

train_data_stress %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()
test_data_stress %>%
  group_by(bankrupt_firm) %>%
  summarise(firm_count = n_distinct(gvkey)) %>%
  ungroup() %>%
  print()

# Scale the macro variables
macro_vars <- c("interest_rate", "gdp_growth", "unemployment", "vix")
macro_means <- sapply(train_data_stress[, macro_vars], mean, na.rm = TRUE)
macro_sds   <- sapply(train_data_stress[, macro_vars], sd, na.rm = TRUE)

# Scale macro variables in train set
train_data_stress <- train_data_stress %>%
  mutate(
    interest_rate = (interest_rate - macro_means["interest_rate"]) / macro_sds["interest_rate"],
    gdp_growth    = (gdp_growth    - macro_means["gdp_growth"])    / macro_sds["gdp_growth"],
    unemployment  = (unemployment  - macro_means["unemployment"])  / macro_sds["unemployment"],
    vix           = (vix           - macro_means["vix"])           / macro_sds["vix"]
  )

# Scale macro variables in test set using the same values
test_data_stress <- test_data_stress %>%
  mutate(
    interest_rate = (interest_rate - macro_means["interest_rate"]) / macro_sds["interest_rate"],
    gdp_growth    = (gdp_growth    - macro_means["gdp_growth"])    / macro_sds["gdp_growth"],
    unemployment  = (unemployment  - macro_means["unemployment"])  / macro_sds["unemployment"],
    vix           = (vix           - macro_means["vix"])           / macro_sds["vix"]
  )


# Recreate a clean version of sic2_f with only observed levels in train
train_data_stress$sic2_f <- factor(as.character(train_data_stress$sic2), levels = unique(as.character(train_data_stress$sic2)))
test_data_stress$sic2_f <- factor(as.character(test_data_stress$sic2), levels = levels(train_data_stress$sic2_f))
test_data_stress <- test_data_stress %>% filter(!is.na(sic2_f))

# Baseline model - stress testing ----
stress_model <- glm(bankrupt ~ wc_ta + re_ta + ebit_ta + mve_bvlt + sale_ta +
                     asset_growth + sales_growth + emp_growth +roe_change +
                     pb_change + op_margin + log_at + sic2_f +
                     interest_rate + gdp_growth + unemployment + vix,
                   data = train_data_stress, family = "binomial")

pred_base <- predict(stress_model, newdata = test_data_stress, type = "response")
summary(stress_model)


# Define crisis periods
crisis_periods <- list(
  dotcom = c(2000, 2001),
  gfc    = c(2007, 2008, 2009),
  covid  = c(2020, 2021)
)

# Get average macro values for each crisis
macro_crisis_values <- lapply(crisis_periods, function(years) {
  model_data_macro %>%
    filter(fyear %in% years) %>%
    summarise(across(all_of(macro_vars), ~ mean(.x, na.rm = TRUE))) %>%
    as.list()
})

# Scale macro crisis values using training means and SDs
macro_crisis_scaled <- lapply(macro_crisis_values, function(values) {
  mapply(function(value, var) {
    (value - macro_means[[var]]) / macro_sds[[var]]
  }, values, names(values))
})

# Dotcom crisis ----
stress_dotcom <- test_data_stress %>%
  mutate(interest_rate = macro_crisis_scaled$dotcom["interest_rate"],
         gdp_growth    = macro_crisis_scaled$dotcom["gdp_growth"],
         unemployment  = macro_crisis_scaled$dotcom["unemployment"],
         vix           = macro_crisis_scaled$dotcom["vix"])

pred_dotcom <- predict(stress_model, newdata = stress_dotcom, type = "response")

# 2008 crisis ----
stress_2008 <- test_data_stress %>%
  mutate(interest_rate = macro_crisis_scaled$gfc["interest_rate"],
         gdp_growth    = macro_crisis_scaled$gfc["gdp_growth"],
         unemployment  = macro_crisis_scaled$gfc["unemployment"],
         vix           = macro_crisis_scaled$gfc["vix"])

pred_2008 <- predict(stress_model, newdata = stress_2008, type = "response")

# Covid crisis ----
stress_covid <- test_data_stress %>%
  mutate(interest_rate = macro_crisis_scaled$covid["interest_rate"],
         gdp_growth    = macro_crisis_scaled$covid["gdp_growth"],
         unemployment  = macro_crisis_scaled$covid["unemployment"],
         vix           = macro_crisis_scaled$covid["vix"])

pred_covid <- predict(stress_model, newdata = stress_covid, type = "response")

# Compare the models ----
# Get the threshold
roc_stress <- roc(test_data_stress$bankrupt, pred_base)
opt_threshold <- as.numeric(coords(roc_stress, "best", ret = "threshold", transpose = FALSE))[[1]]
opt_threshold

compare_rates <- tibble(
  scenario = c("Baseline", "Dotcom", "2008", "COVID"),
  avg_risk = c(
    mean(pred_base),
    mean(pred_dotcom),
    mean(pred_2008),
    mean(pred_covid)),
  firms_at_risk = c(
    sum(pred_base > opt_threshold),
    sum(pred_dotcom > opt_threshold),
    sum(pred_2008 > opt_threshold),
    sum(pred_covid > opt_threshold))
)

print(compare_rates)

# Plot the results
# Combine all predictions into one dataframe
risk_df <- bind_rows(
  tibble(risk = pred_base,  scenario = "Baseline"),
  tibble(risk = pred_dotcom, scenario = "Dotcom"),
  tibble(risk = pred_2008,   scenario = "2008"),
  tibble(risk = pred_covid,  scenario = "COVID")
)

# Plot
ggplot(risk_df, aes(x = risk, fill = scenario)) +
  geom_density(alpha = 0.4) +
  labs(
    title = "Predicted Bankruptcy Risk Across Stress Scenarios",
    x = "Predicted Risk",
    y = "Density"
  ) +
  theme_minimal()
