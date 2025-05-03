##### 0.  Libraries
-----------------------------------------------------------
library(glmnet)
library(randomForest)
library(xgboost)
library(rstanarm)
library(BoomSpikeSlab)
library(caret)
library(MASS)
library(Matrix)
pkgs <- c("glmnet", "dplyr", "tidyr", "purrr", "ggplot2")
invisible(lapply(pkgs[!(pkgs %in% installed.packages())], install.packages))
invisible(lapply(pkgs, library, character.only = TRUE))
`%nin%` <- Negate(`%in%`)          # convenience for set complements
set.seed(20250426)                 # reproducibility

##### 1.  Global settings -----------------------------------------------------
n       <- 100          # rows
p       <- 100          # columns
B       <- 30           # number of independent simulations
truth   <- list(        # ground-truth active predictors for each response
  Y1 = c("X1","X2","X3","X4","X5"),
  Y2 = c("X1","X2","X3","X4","X5","X1X2","X3X4"),
  Y3 = c("X1","X2","X3","X4","X5","X3_sq","X4_sq")
)

##### 2.  Helper that fits three models and tallies TP/TN/FP/FN ---------------
score_one <- function(y, xmat, predictors, truth_set){
  ## Poisson GLM – select by p value
  form      <- as.formula(paste(y, "~", paste(predictors, collapse = "+")))
  fit_glm   <- glm(form, family = poisson(), data = xmat)
  sel_glm   <- names(coef(summary(fit_glm)))[which(coef(summary(fit_glm))[-1,"Pr(>|z|)"] < .05)]
  
  ## LASSO (alpha = 1)
  cv_lasso  <- cv.glmnet(as.matrix(xmat[,predictors]), xmat[[y]], family = "poisson", alpha = 1)
  sel_lasso <- rownames(coef(cv_lasso, s = "lambda.min"))[coef(cv_lasso, s = "lambda.min")[,1] != 0][-1]
  
  ## Elastic Net (alpha = .5)
  cv_en     <- cv.glmnet(as.matrix(xmat[,predictors]), xmat[[y]], family = "poisson", alpha = .5)
  sel_en    <- rownames(coef(cv_en, s = "lambda.min"))[coef(cv_en, s = "lambda.min")[,1] != 0][-1]
  
  all_sets  <- list(GLM = sel_glm, LASSO = sel_lasso, ElasticNet = sel_en)
  purrr::imap_dfr(all_sets, \(sel, m){
    tibble(method = m,
           TP = sum(sel %in% truth_set),
           FP = sum(sel %nin% truth_set),
           FN = sum(truth_set %nin% sel),
           TN = length(setdiff(predictors, union(sel, truth_set))))
  })
}

##### 3.  Main simulation loop -------------------------------------------------
results <- purrr::map_dfr(1:B, function(b){
  ## -- simulate predictors
  X             <- matrix(rnorm(n*p), n, p, dimnames = list(NULL, paste0("X", 1:p)))
  df            <- as.data.frame(X)
  ## -- derived interaction / quadratic columns
  df$X1X2       <- df$X1 * df$X2
  df$X3X4       <- df$X3 * df$X4
  df$X3_sq      <- df$X3^2
  df$X4_sq      <- df$X4^2
  
  ## -- linear predictors
  linpred <- within(df, {
    Y1_lp <- 0.5*X1 + 0.5*X2 + 0.5*X3 + 0.5*X4 + 0.5*X5
    Y2_lp <- Y1_lp + 0.5*X1X2 + 0.5*X3X4
    Y3_lp <- Y1_lp +      X3_sq +      X4_sq
  })
  
  ## -- turn into Poisson counts
  df$Y1 <- rpois(n, lambda = exp(linpred$Y1_lp))
  df$Y2 <- rpois(n, lambda = exp(linpred$Y2_lp))
  df$Y3 <- rpois(n, lambda = exp(linpred$Y3_lp))
  
  predictors <- setdiff(names(df), c("Y1","Y2","Y3"))
  
  ## -- fit & score for each response
  purrr::imap_dfr(list(Y1 = "Y1", Y2 = "Y2", Y3 = "Y3"), \(resp, nm){
    score_one(resp, df, predictors, truth[[nm]]) |>
      mutate(sim = b, response = nm)
  })
})

##### 4.  Summary table --------------------------------------------------------
summary_tbl <- results |>
  group_by(response, method) |>
  summarise(across(c(TP,TN,FP,FN),
                   list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
            .groups = "drop")
print(summary_tbl)

##### 5.  Box-plots ------------------------------------------------------------
results_long <- results |>
  tidyr::pivot_longer(cols = TP:FN, names_to = "metric", values_to = "value")

ggplot(results_long, aes(method, value)) +
  geom_boxplot(outlier.shape = NA, width = .6) +
  facet_grid(metric ~ response, scales = "free_y") +
  theme_bw(base_size = 11) +
  labs(title = "Variable-selection performance (30 simulated data sets)",
       x = NULL, y = "Count")

# A tibble: 9 × 10
#response method TP_mean TN_mean FP_mean FN_mean TP_sd …  
#
## ---------------------------------------------------------------------------
##  Heat-map of mean TP / TN / FP / FN counts
## ---------------------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(viridis)   # nice perceptually-uniform palette

heat_df <- summary_tbl %>%                       # <-- object created earlier
  select(response, method, ends_with("_mean")) %>%           # keep means only
  pivot_longer(cols  = ends_with("_mean"),
               names_to  = "metric",
               values_to = "mean") %>%
  mutate(metric = factor(metric,
                         levels = c("TP_mean","TN_mean","FP_mean","FN_mean"),
                         labels = c("TP","TN","FP","FN")))

ggplot(heat_df, aes(method, metric, fill = mean)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(aes(label = round(mean, 1)), size = 3) +      # numeric labels
  facet_wrap(~response, nrow = 1) +
  scale_fill_viridis_c(option = "A", name = "Mean\ncount") +
  labs(title = "Variable-selection performance (30 simulations)",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.spacing = unit(1, "lines"))




#######normalvslaplacepriors
###############################################################################
## 0.  Libraries -------------------------------------------------------------
###############################################################################
pkgs <- c("rstanarm", "dplyr", "tidyr", "purrr", "ggplot2", "tibble",
          "stringr", "furrr", "posterior")
invisible(lapply(pkgs[!(pkgs %in% installed.packages())], install.packages))
invisible(lapply(pkgs, library, character.only = TRUE))

options(mc.cores = max(1, parallel::detectCores() - 1))
rstan_options(auto_write = TRUE)

set.seed(123)           # reproduce everything

###############################################################################
## 1.  Global settings --------------------------------------------------------
###############################################################################
n_sims   <- 30     # <-- reduce while prototyping
n        <- 100
p        <- 100
sigma_y  <- 1      # SD of Gaussian noise; set to 0 for noise-free

###############################################################################
## 2.  Helper: generate one data frame ---------------------------------------
###############################################################################
make_dataset <- function(id, n, p, sigma = 1) {
  X          <- matrix(rnorm(n * p), n, p,
                       dimnames = list(NULL, paste0("X", 1:p)))
  X_df       <- as_tibble(X)
  
  ## Derived predictors used in Y2 and Y3 -----------------
  X_df <- X_df %>%
    mutate(X1_X2 = X1 * X2,
           X3_X4 = X3 * X4,
           X3_sq = X3^2,
           X4_sq = X4^2)
  
  ## Linear predictors ------------------------------------
  eta1 <- 0.5*(X_df$X1 + X_df$X2 + X_df$X3 + X_df$X4 + X_df$X5)
  eta2 <- eta1 + 0.5*X_df$X1_X2 + 0.5*X_df$X3_X4
  eta3 <- eta1 +       X_df$X3_sq +       X_df$X4_sq
  
  ## Responses (+ optional noise) -------------------------
  Y1 <- eta1 + rnorm(n, 0, sigma)
  Y2 <- eta2 + rnorm(n, 0, sigma)
  Y3 <- eta3 + rnorm(n, 0, sigma)
  
  X_df %>% mutate(Y1 = Y1, Y2 = Y2, Y3 = Y3,  .dataset = id)
}

###############################################################################
## 3.  Truth vectors (for confusion matrix) -----------------------------------
###############################################################################
truth_vars <- list(
  Y1 = c("X1","X2","X3","X4","X5"),
  Y2 = c("X1","X2","X3","X4","X5","X1_X2","X3_X4"),
  Y3 = c("X1","X2","X3","X4","X5","X3_sq","X4_sq")
)

all_predictors <- c(paste0("X", 1:p), "X1_X2","X3_X4","X3_sq","X4_sq")
total_p        <- length(all_predictors)

###############################################################################
## 4.  One-fit function: returns TP/TN/FP/FN ----------------------------------
###############################################################################
fit_and_score <- function(df, response, prior_type = c("normal","laplace")) {
  prior_type  <- match.arg(prior_type)
  form        <- as.formula(paste(response, "~", paste(all_predictors, collapse = "+")))
  
  fit <- stan_glm(
    form, data = df,
    prior = if (prior_type == "normal")  normal() else laplace(),
    prior_intercept = normal(),  # weakly informative on α
    chains = 1, iter = 500, seed = 123, refresh = 0)
  
  # 95 % posterior intervals
  post_int <- posterior_interval(fit, prob = 0.95)
  sel      <- !(post_int[,1] < 0 & post_int[,2] > 0)   # TRUE = selected
  sel      <- sel[names(sel) != "(Intercept)"]          # drop intercept
  
  true_set <- truth_vars[[response]]
  pred_set <- names(sel)[sel]
  
  TP <- sum(pred_set %in%  true_set)
  FP <- sum(pred_set %notin% true_set)
  FN <- sum(setdiff(true_set, pred_set) %in% true_set)
  TN <- total_p - TP - FP - FN
  
  tibble(dataset   = df$.dataset[1],
         response  = response,
         prior     = prior_type,
         TP, TN, FP, FN,
         TPR = TP / (TP + FN),
         TNR = TN / (TN + FP),
         FPR = FP / (FP + TN),
         FNR = FN / (TP + FN))
}

`%notin%` <- Negate(`%in%`)

###############################################################################
## 5.  Run the full simulation -----------------------------------------------
###############################################################################
plan(multisession, workers = max(1, parallel::detectCores() - 1))

results <- future_map_dfr(1:n_sims, function(i){
  dat_i <- make_dataset(i, n, p, sigma = sigma_y)
  map_dfr(c("Y1","Y2","Y3"), ~{
    bind_rows(
      fit_and_score(dat_i, .x, "normal"),
      fit_and_score(dat_i, .x, "laplace")
    )
  })
}, .progress = TRUE)

###############################################################################
## 6.  Summary tables ---------------------------------------------------------
###############################################################################
summary_tbl <- results %>%
  group_by(response, prior) %>%
  summarise(across(c(TP,TN,FP,FN,TPR,TNR,FPR,FNR),
                   list(mean = mean, sd = sd), .names="{.col}_{.fn}"),
            .groups = "drop")

print(summary_tbl, n = Inf)

###############################################################################
## 7.  Figures ----------------------------------------------------------------
###############################################################################
## (a) Box-plots of TPR and FPR by prior
metrics_long <- results %>%
  pivot_longer(cols = c(TPR, FPR), names_to = "metric", values_to = "value")

ggplot(metrics_long,
       aes(prior, value, fill = prior)) +
  geom_boxplot(alpha = 0.8) +
  facet_grid(metric ~ response, scales = "free_y") +
  labs(title = "Normal vs Laplace prior – variable-selection performance",
       x = NULL, y = "Rate") +
  theme_bw() +
  theme(legend.position = "none")

## (b) Heat-map of average confusion counts
conf_heat <- results %>%
  group_by(response, prior) %>%
  summarise(across(c(TP,TN,FP,FN), mean), .groups="drop") %>%
  pivot_longer(cols = TP:FN, names_to = "count", values_to = "mean")

ggplot(conf_heat,
       aes(count, prior, fill = mean)) +
  geom_tile() +
  facet_wrap(~response) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Average confusion-matrix counts",
       x = NULL, y = NULL) +
  theme_minimal()





#############spike-and-lab prior


################################################################################
## 0.  Libraries --------------------------------------------------------------
################################################################################
pkgs <- c("BoomSpikeSlab", "dplyr", "purrr", "tidyr", "tibble",
          "ggplot2", "pheatmap")
invisible(lapply(pkgs[!(pkgs %in% installed.packages())], install.packages))
invisible(lapply(pkgs, library, character.only = TRUE))

set.seed(20250427)                     # reproducibility

################################################################################
## 1.  Global settings --------------------------------------------------------
################################################################################
n_sims     <- 30                       # number of simulated data sets
n          <- 100                      # rows per data set
p          <- 100                      # base-matrix predictors
pip_cutoff <- 0.50                     # selection threshold
burn_in    <- 2500                     # draws discarded when summarising

# “signal” variables ----------------------------------------------------------
true_vars <- list(
  Y1 = 1:5,
  Y2 = c(1:5, 101, 102),               # X1*X2 , X3*X4
  Y3 = c(1:5, 101, 102)                # X3^2  , X4^2
)

################################################################################
## 2.  Helper functions -------------------------------------------------------
################################################################################
simulate_X <- function(n, p) {
  matrix(rnorm(n * p), nrow = n, ncol = p,
         dimnames = list(NULL, paste0("X", seq_len(p))))
}

generate_Y <- function(X) {
  X_ext <- cbind(
    X,
    X[, 1] * X[, 2],                   # col 101
    X[, 3] * X[, 4]                    # col 102
  )
  colnames(X_ext)[(p + 1):(p + 2)] <- c("X1X2_or_X3sq", "X3X4_or_X4sq")
  
  with(as.data.frame(X_ext), {
    eta1 <- 0.5 * (X1 + X2 + X3 + X4 + X5)
    eta2 <- eta1 + 0.5 * X1X2_or_X3sq + 0.5 * X3X4_or_X4sq
    eta3 <- eta1 +       X1X2_or_X3sq +       X3X4_or_X4sq
    list(
      Y1 = eta1 + rnorm(n),
      Y2 = eta2 + rnorm(n),
      Y3 = eta3 + rnorm(n),
      X_ext = X_ext
    )
  })
}

fit_spike_slab <- function(X, y, niter = 5000) {
  lm.spike(
    y ~ ., data = as.data.frame(X),
    niter = niter,
    expected.model.size = 8
  )
}

# ---- NEW, robust inclusion-probability extractor ----------------------------
get_pip_safe <- function(fit, p_total, burn = burn_in) {
  
  s <- summary(fit, burn = burn)
  
  pip <- NULL
  # 1.  Current BoomSpikeSlab (≥ 1.2.x)
  if (!is.null(s$coefficients) &&
      "inc.prob" %in% colnames(s$coefficients)) {
    pip <- s$coefficients[, "inc.prob"]
    names(pip) <- rownames(s$coefficients)
  }
  
  # 2.  Very old versions (posterior.inclusion.probabilities / posterior.prob)
  if (is.null(pip)) {
    pip <- s$posterior.inclusion.probabilities
  }
  if (is.null(pip)) {
    pip <- s$posterior.prob
  }
  if (is.null(pip)) {
    stop("Cannot extract posterior inclusion probabilities.")
  }
  
  # Drop intercept if present -----------------------------------------------
  pip <- pip[names(pip) != "(Intercept)"]
  
  # Sanity check -------------------------------------------------------------
  if (length(pip) != p_total) {
    stop("PIP length (", length(pip),
         ") does not match number of predictors (", p_total, ").")
  }
  
  unname(pip)
}

metrics_from_selection <- function(selected, truth, p_total) {
  tp <- sum(selected &  truth)
  fp <- sum(selected & !truth)
  fn <- sum(!selected &  truth)
  tn <- p_total - tp - fp - fn
  tibble(
    TP  = tp, FP = fp, FN = fn, TN = tn,
    TPR = tp / (tp + fn),
    TNR = tn / (tn + fp),
    FPR = fp / (fp + tn),
    FNR = fn / (fn + tp)
  )
}

################################################################################
## 3.  Main simulation loop ---------------------------------------------------
################################################################################
results_list <- list()     # rate metrics
sel_list     <- list()     # 0/1 selections for heat-map

for (sim_id in seq_len(n_sims)) {
  
  X      <- simulate_X(n, p)
  y_list <- generate_Y(X)
  X_full <- y_list$X_ext                    # 100 × 102
  
  for (resp_id in 1:3) {
    
    resp_name <- paste0("Y", resp_id)
    y         <- y_list[[resp_name]]
    X_use     <- if (resp_id == 1) X else X_full
    p_total   <- ncol(X_use)
    
    fit       <- fit_spike_slab(X_use, y)
    pip       <- get_pip_safe(fit, p_total)
    selected  <- pip > pip_cutoff
    
    truth_vec <- rep(FALSE, p_total)
    truth_vec[ true_vars[[resp_name]] ] <- TRUE
    
    # ---- metrics -----------------------------------------------------------
    results_list[[length(results_list) + 1]] <-
      metrics_from_selection(selected, truth_vec, p_total) |>
      mutate(sim = sim_id, response = resp_name)
    
    # ---- selections for heat-map -------------------------------------------
    sel_list[[length(sel_list) + 1]] <-
      tibble(
        sim      = sim_id,
        response = resp_name,
        variable = seq_len(p_total),
        selected = selected
      )
  }
}

results <- bind_rows(results_list)
sel_tbl <- bind_rows(sel_list)

################################################################################
## 4.  Summary table ----------------------------------------------------------
################################################################################
metrics_summary <- results |>
  pivot_longer(c(TP:TNR, FPR:FNR), names_to = "metric") |>
  group_by(response, metric) |>
  summarise(mean = mean(value), sd = sd(value), .groups = "drop") |>
  arrange(response, metric)

print(metrics_summary)

################################################################################
## 5.  Box-plots --------------------------------------------------------------
################################################################################
metrics_long <- results |>
  pivot_longer(c(TPR, TNR, FPR, FNR), names_to = "metric")

ggplot(metrics_long,
       aes(x = metric, y = value)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ response, nrow = 1) +
  labs(title = "Spike–and–Slab Variable-Selection Rates (30 runs)",
       x = NULL, y = "Rate") +
  theme_minimal(base_size = 13)

################################################################################
## 6.  Heat-map ---------------------------------------------------------------
################################################################################
heat_df <- sel_tbl |>
  group_by(response, variable) |>
  summarise(freq = mean(selected), .groups = "drop")

heat_mat <- heat_df |>
  pivot_wider(names_from = response, values_from = freq, values_fill = 0) |>
  arrange(variable) |>
  column_to_rownames("variable") |>
  as.matrix()

pheatmap(
  heat_mat,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  color        = colorRampPalette(c("white", "steelblue4"))(100),
  main         = "Variable-selection heat-map\n(proportion of 30 simulations)",
  fontsize_row = 6
)





####################ml methods

###############################################################################
## 0.  Libraries -------------------------------------------------------------
###############################################################################
pkgs <- c("randomForest", "xgboost",
          "dplyr", "tidyr", "purrr", "tibble",
          "ggplot2", "forcats")
invisible(lapply(pkgs[!(pkgs %in% installed.packages())], install.packages))
invisible(lapply(pkgs, library, character.only = TRUE))

set.seed(20250426)

###############################################################################
## 1.  Settings ---------------------------------------------------------------
###############################################################################
n_sims      <- 30        # number of simulated data sets
n           <- 100       # rows per data set
p           <- 100       # columns (= predictors)
true_var    <- 1:5       # truly relevant predictors
ntree_rf    <- 500       # #trees for Random Forest
nrounds_xgb <- 200       # boosting iterations for XGBoost

###############################################################################
## 2.  Helper functions -------------------------------------------------------
###############################################################################
make_Y <- function(X) {
  x1 <- X[, 1]; x2 <- X[, 2]; x3 <- X[, 3]; x4 <- X[, 4]; x5 <- X[, 5]
  eps <- rnorm(nrow(X))
  list(
    Y1 = 0.5 * (x1 + x2 + x3 + x4 + x5)                      + eps,
    Y2 = 0.5 * (x1 + x2 + x3 + x4 + x5 + x1 * x2 + x3 * x4)  + eps,
    Y3 = 0.5 * (x1 + x2 + x3 + x4 + x5) + x3^2 + x4^2        + eps
  )
}

get_conf <- function(sel, p, truth = true_var) {
  TP <- sum(sel %in% truth)
  FP <- length(sel) - TP
  TN <- (p - length(truth)) - FP
  FN <- length(truth) - TP
  c(TP = TP, TN = TN, FP = FP, FN = FN)
}

###############################################################################
## 3.  Simulation loop --------------------------------------------------------
###############################################################################
results_metrics   <- list()
results_selection <- list()

for (s in 1:n_sims) {
  set.seed(1000 + s)                                # make sims independent
  X  <- matrix(rnorm(n * p), n, p,
               dimnames = list(NULL, paste0("V", 1:p)))
  Ys <- make_Y(X)                                   # list with Y1, Y2, Y3
  
  for (resp in names(Ys)) {
    y <- Ys[[resp]]
    
    ## tryCatch so a failure in one combo does not halt everything -----------
    tryCatch({
      ## ---------------- Random Forest --------------------------------------
      rf_fit <- randomForest(x = X, y = y,
                             ntree = ntree_rf, importance = TRUE)
      rf_imp <- importance(rf_fit, type = 1)[, 1]
      rf_top <- order(rf_imp, decreasing = TRUE)[1:10]
      
      ## ------------------- XGBoost -----------------------------------------
      dmat <- xgb.DMatrix(data = X, label = y)       # build DM
      colnames(dmat) <- colnames(X)                  # attach feature names
      
      xgb_fit <- xgb.train(params  = list(objective = "reg:squarederror",
                                          eval_metric = "rmse"),
                           data    = dmat,
                           nrounds = nrounds_xgb,
                           verbose = 0)
      
      xgb_imp  <- xgb.importance(feature_names = colnames(X),
                                 model = xgb_fit)
      xgb_vars <- as.integer(sub("V", "", xgb_imp$Feature))
      if (length(xgb_vars) < 10)                     # pad to 10 if needed
        xgb_vars <- c(xgb_vars,
                      sample(setdiff(1:p, xgb_vars),
                             10 - length(xgb_vars)))
      xgb_top <- xgb_vars[1:10]
      
      ## ---------------- store metrics & selections -------------------------
      for (mdl in c("RF", "XGB")) {
        top_vars <- if (mdl == "RF") rf_top else xgb_top
        conf <- as.list(get_conf(top_vars, p))   # TP, TN, FP, FN
        results_metrics[[length(results_metrics) + 1]] <-
          tibble(sim      = s,
                 response = resp,
                 method   = mdl,
                 !!!conf)                          # splice the four named entries
        results_selection[[length(results_selection) + 1]] <-
          tibble(sim = s, response = resp, method = mdl,
                 variable = top_vars)
      }
    }, error = function(e) {
      message("⚠️  Skipped sim ", s, ", response ", resp, ": ", e$message)
    })
  }
}

###############################################################################
## 4.  Assemble tidy result objects ------------------------------------------
###############################################################################
## -- ensure the objects always exist ----------------------------------------
metrics_df   <- tibble()   # <- guarantees the name exists
selection_df <- tibble()

## -- did at least one replicate succeed? ------------------------------------
if (length(results_metrics) == 0) {
  stop("No simulation completed successfully – see earlier warnings.")
}

## -- build the two big data frames ------------------------------------------
metrics_df   <- bind_rows(results_metrics) |>
  mutate(across(where(is.character), as.factor))

selection_df <- bind_rows(results_selection)

###############################################################################
## 5.  Summary tables ---------------------------------------------------------
###############################################################################
## -- same safety check for the summary --------------------------------------
if (nrow(metrics_df) == 0) {
  stop("metrics_df is empty – cannot create summary_tbl.")
}

summary_tbl <- metrics_df |>
  group_by(response, method) |>
  summarise(across(c(TP, TN, FP, FN),
                   list(mean = mean, sd = sd),
                   .names = "{.col}_{.fn}"),
            .groups = "drop")

print(summary_tbl)


###############################################################################
## 6.  Figures ---------------------------------------------------------------
###############################################################################
## (a) Box-plots for TP, TN, FP, FN
metrics_long <- metrics_df |>
  pivot_longer(cols = TP:FN, names_to = "metric", values_to = "value")

ggplot(metrics_long,
       aes(x = method, y = value, fill = method)) +
  geom_boxplot(width = 0.7, alpha = 0.7, outlier.shape = NA) +
  facet_grid(metric ~ response, scales = "free_y") +
  labs(title = "Distribution of confusion-matrix metrics (30 replicates)",
       x = NULL, y = NULL) +
  theme_bw() +
  theme(legend.position = "none")

## (b) Compute frequency table for heatmap
freq_tbl <- selection_df |>
  count(response, method, variable) |>
  group_by(response, method) |>
  mutate(freq = n / n_sims) |>
  ungroup()

## (b) Heat-map: frequency a variable entered top-10 (first 20 predictors)
heatmap_df <- freq_tbl |>
  filter(variable <= 20)

ggplot(heatmap_df,
       aes(x = factor(variable),
           y = interaction(response, method, lex.order = TRUE),
           fill = freq)) +
  geom_tile() +
  scale_fill_continuous(name = "Proportion\nof runs") +
  labs(x = "Predictor index",
       y = "Response × Model",
       title = "Top-10 selection frequency (variables 1–20 shown)") +
  theme_bw()

###############################################################################
##############Real Data Analysis
##################
###################
##############Real Data Analysis
##################
###################
#selecting the file containing the dataset
filteredrdata <- file.choose()


###############OR


# Load it back later
filteredrdata <- readRDS("filteredrdata.rds")



# Define predictors and response
X <- model.matrix(deaths ~ . -1, data = filteredrdata)
y <- filteredrdata$deaths

# Split into training and testing sets
set.seed(4533)
train_index <- createDataPartition(y, p = 0.5, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]




###############################
##########################
###############11111111111111111
###################POISSON
#############################
#############################
poisson_model <- glm(deaths ~ ., family = poisson(link = "log"),
                     data = filteredrdata[train_index, ])


step_model <-step(poisson_model, direction = "both")


summary(step_model)










#####################################222222AND3333333
####################222222AND3333333. LASSO and ENET
#########################################
##########################
library(glmnet)

# Assuming X_train is your predictor matrix and y_train is your response vector
# For Lasso (alpha = 1)
lasso_model <- glmnet(X_train, y_train, alpha = 1)

# For Elastic Net (e.g., alpha = 0.5)
enet_model <- glmnet(X_train, y_train, alpha = 0.5)




par(mfrow = c(1, 1))
# Plot for Lasso
plot(lasso_model, xvar = "lambda", label = TRUE, main = "Lasso Regularization Path")

# Plot for Elastic Net
plot(enet_model, xvar = "lambda", label = TRUE, main = "Elastic Net Regularization Path")



# Cross-validation for Lasso
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)
plot(cv_lasso)

# Cross-validation for Elastic Net
cv_enet <- cv.glmnet(X_train, y_train, alpha = 0.5)
plot(cv_enet)



# Coefficients for Lasso at optimal λ
coef(cv_lasso, s = "lambda.min")

# Coefficients for Elastic Net at optimal λ
coef(cv_enet, s = "lambda.min")





# For Lasso
lasso_coef <- coef(cv_lasso, s = "lambda.min")
lasso_selected <- rownames(lasso_coef)[lasso_coef[, 1] != 0]

# For Elastic Net
enet_coef <- coef(cv_enet, s = "lambda.min")
enet_selected <- rownames(enet_coef)[enet_coef[, 1] != 0]




# Predictions
lasso_pred <- predict(cv_lasso, newx = X_test, s = "lambda.min")
enet_pred <- predict(cv_enet, newx = X_test, s = "lambda.min")

# Compute RMSE
lasso_rmse <- sqrt(mean((y_test - lasso_pred)^2))
enet_rmse <- sqrt(mean((y_test - enet_pred)^2))



# Cross-validated RMSE
lasso_cv_rmse <- min(cv_lasso$cvm)
enet_cv_rmse <- min(cv_enet$cvm)






# Fit Poisson GLM
glm_model <- glm(y_train ~ ., family = "poisson", data = X_train)

# Fit Lasso
library(glmnet)
x_train_matrix <- model.matrix(y_train ~ ., train_data)[, -1]
lasso_model <- cv.glmnet(x_train_matrix, y_train, family = "poisson", alpha = 1)

# Fit Elastic Net
enet_model <- cv.glmnet(x_train_matrix, y_train, family = "poisson", alpha = 0.5)

# Extract non-zero coefficients
lasso_coef <- coef(lasso_model, s = "lambda.min")
enet_coef <- coef(enet_model, s = "lambda.min")

# Identify selected variables
lasso_selected <- rownames(lasso_coef)[lasso_coef[, 1] != 0]
enet_selected <- rownames(enet_coef)[enet_coef[, 1] != 0]




####################4. Random Forest Regression
#########################################
##########################
# Fit the random forest model
set.seed(453453)
#sample_indices <- sample(1:nrow(train_data), size = 1000)
#sampled_data <- train_data[sample_indices, ]
#rf_model <- randomForest(deaths ~ ., data = sampled_data)

rf_model <- randomForest(deaths ~ ., data = X_train, importance = TRUE)

# Display variable importance
importance(rf_model)

# Plot variable importance
varImpPlot(rf_model)

##############
###################
# Fit Random Forest model

rf_model <- randomForest(x = X_train, y = y_train, ntree = 2)

# Predict on test data
y_pred_rf <- predict(rf_model, newdata = X_test)



# Display variable importance
importance(rf_model)

# Plot variable importance
varImpPlot(rf_model)
summary(rf_model)

plot(rf_model)


library(randomForest)
rf_model2 <- randomForest(x = X_train, y = y_train, importance = TRUE)



##############################
############################
#########################
#######################
####################5
#########################################
##########################
##############
###################
# Prepare data for XGBoost
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

# Set parameters
params <- list(objective = "reg:squarederror", eval_metric = "rmse")

# Train XGBoost model

# Define watchlist to monitor training and validation error
#watchlist <- list(train = dtrain, eval = dtest)

# Train the model with early stopping
#xgb_model <- xgb.train(
#  params = params,
#  data = dtrain,
# nrounds = 200,
#watchlist = watchlist,
#  early_stopping_rounds = 10,
#  print_every_n = 10
#)


xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 10)



importance_matrix <- xgb.importance(model = xgb_model)

importance_matrix <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)


xgb.plot.importance(importance_matrix)
print(importance_matrix)
##############################
############################
#########################
####












##############################
##############################
############################
#########################
#######################
####################6 Fit Bayesian regression model with normal prior
#########################################
##########################
##############
###################


bayesian_normal_model <- stan_glm(deaths ~ ., data = filteredrdata[train_index, ],
                                  family = gaussian(), prior = normal(0, 1),
                                  prior_intercept = normal(0, 1), chains = 4,
                                  iter = 1000)

# Predict on test data
y_pred_bayes_normal <- posterior_predict(bayesian_normal_model, 
                                         newdata = filteredrdata[-train_index, ])
# Use mean predictions
y_pred_bayes_normal_mean <- rowMeans(y_pred_bayes_normal)



summary(bayesian_normal_model)



plot(bayesian_normal_model)

##############################
############################
#########################
#######################
####################7
#########################################
##########################
##############
###################
# Fit Bayesian regression model with Laplace (Lasso) prior
bayesian_laplace_model <- stan_glm(deaths ~ .,
                                   data = filteredrdata[train_index, ],
                                   family = gaussian(), 
                                   prior = laplace(0, 1), 
                                   prior_intercept = normal(0, 1),
                                   chains = 4, iter = 1000)

# Predict on test data
y_pred_bayes_laplace <- posterior_predict(bayesian_laplace_model,
                                          newdata = filteredrdata[-train_index, ])
# Use mean predictions
y_pred_bayes_laplace_mean <- rowMeans(y_pred_bayes_laplace)



summary(bayesian_laplace_model)


plot(bayesian_laplace_model)
##############################
############################
#########################
#######################
####################8
#########################################
##########################
##############
###################


library(spikeslab)
# Combine training data
train_data <- filteredrdata[train_index, ]

# Fit Spike-and-Slab model

spike_slab_model <- spikeslab(deaths ~ ., data = train_data)

# Predict on test data
test_data <- filteredrdata[-train_index, ]
#y_pred_spike_slab <- predict(spike_slab_model, newdata = test_data)

# Obtain predictions from the spikeslab model
predictions <- predict(spike_slab_model, newdata = test_data)

# Extract the 'bma' component, which is a numeric vector
y_pred_spike_slab <- predictions$bma



########################
# Install the BoomSpikeSlab package
install.packages("BoomSpikeSlab")

# Load the package
library(BoomSpikeSlab)

# Prepare your predictor matrix (X) and response vector (y)
X <- model.matrix(deaths ~ ., data = train_data)[, -1]  # Remove intercept
y <- train_data$deaths

# Fit the spike-and-slab model
model <- lm.spike(y ~ X, niter = 10)

# Summarize the model with a burn-in of 100 iterations
summary(model, burn = 100)

# Fit the model with Student-t errors
model <- lm.spike(deaths ~ ., data = train_data, niter = 1000, error.distribution = "student")
model2 <- lm.spike(deaths ~ ., data = train_data, niter = 500, error.distribution = "student")

summary(model)
plot(model)
summary(model2)
plot(model2)
##############################
##################