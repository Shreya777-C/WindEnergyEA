#EA Final Project
#Shreya Chennapragada


pkgs <- c(
  "tidyverse", "lubridate", "readxl",
  "forecast", "tseries",
  "ranger", "xgboost"
)

to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

library(tidyverse)
library(lubridate)
library(readxl)
library(forecast)
library(tseries)
library(ranger)
library(xgboost)


PATH_CSV  <- "C:/Users/shrey/Downloads/eia_us_wind_monthly_2001_2025.csv"


load_from_csv <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  
  req_cols <- c("Month", "Wind_Generation_Thousand_MWh")
 
  
  df %>%
    transmute(
      date = as.Date(paste0(Month, "-01")),
      wind = as.numeric(Wind_Generation_Thousand_MWh)
    ) %>%
    arrange(date)
}


wind_df <- load_from_csv(PATH_CSV)

#----------------Task 2
cat("\n Missing Values \n")
cat(" date:", sum(is.na(wind_df$date)), "\n")
cat("wind:", sum(is.na(wind_df$wind)), "\n")

wind_df <- wind_df %>% filter(!is.na(date), !is.na(wind))
cat("Rows after deletion:", nrow(wind_df), "\n")


#Task 1 
ggplot(wind_df, aes(date, wind)) +
  geom_line() +
  labs(title = "U.S. Monthly Wind Generation (EIA)", x = "Date", y = "Wind (Thousand MWh)") +
  theme_minimal() %>%
  print()


start_y <- year(min(wind_df$date))
start_m <- month(min(wind_df$date))
y_ts <- ts(wind_df$wind, start = c(start_y, start_m), frequency = 12)

#STL decomp
cat("\n STL Decomposition \n")
stl_fit <- stl(y_ts, s.window = "periodic")
plot(stl_fit, main = "STL Decomposition (Wind)")

#Box Cox

lambda <- BoxCox.lambda(y_ts, method = "guerrero")
cat("Box-Cox:", lambda, "\n")


#Train Test Split
test_h <- 24
n <- nrow(wind_df)
if (n <= test_h + 36) 
train_df <- wind_df[1:(n - test_h), ]
test_df  <- wind_df[(n - test_h + 1):n, ]

train_ts <- ts(train_df$wind,
               start = c(year(min(train_df$date)), month(min(train_df$date))),
               frequency = 12)

test_ts_truth <- test_df$wind

# Metrics 
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))
mae  <- function(y, yhat) mean(abs(y - yhat), na.rm = TRUE)
mape <- function(y, yhat) mean(abs((y - yhat) / y), na.rm = TRUE) * 100

#-------------Task 5


naive_fc  <- naive(train_ts,  h = test_h)$mean
snaive_fc <- snaive(train_ts, h = test_h)$mean
drift_fc  <- rwf(train_ts, drift = TRUE, h = test_h)$mean
# ETS and ARIMA
ets_fit <- ets(train_ts)
arima_fit <- auto.arima(train_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)

ets_fc_test   <- forecast(ets_fit,   h = test_h)$mean
arima_fc_test <- forecast(arima_fit, h = test_h)$mean

task5_tbl <- tibble(
  model = c("Naive", "SeasonalNaive", "Drift", "ETS", "ARIMA"),
  RMSE  = c(rmse(test_ts_truth, naive_fc),
            rmse(test_ts_truth, snaive_fc),
            rmse(test_ts_truth, drift_fc),
            rmse(test_ts_truth, ets_fc_test),
            rmse(test_ts_truth, arima_fc_test)),
  MAE   = c(mae(test_ts_truth, naive_fc),
            mae(test_ts_truth, snaive_fc),
            mae(test_ts_truth, drift_fc),
            mae(test_ts_truth, ets_fc_test),
            mae(test_ts_truth, arima_fc_test)),
  MAPE  = c(mape(test_ts_truth, naive_fc),
            mape(test_ts_truth, snaive_fc),
            mape(test_ts_truth, drift_fc),
            mape(test_ts_truth, ets_fc_test),
            mape(test_ts_truth, arima_fc_test))
) %>% arrange(RMSE)

print(task5_tbl)

#-------Task 6

# 6.1 
cat("\nADF test on training series:\n")
print(adf.test(train_ts))

d1 <- ndiffs(train_ts)
D1 <- nsdiffs(train_ts)
cat("\nSuggested differencing: d =", d1, " seasonal D =", D1, "\n")

# 6.2 
diff_ts <- train_ts
if (D1 > 0) diff_ts <- diff(diff_ts, lag = 12, differences = D1)
if (d1 > 0) diff_ts <- diff(diff_ts, differences = d1)

acf(diff_ts, main = "ACF ")
pacf(diff_ts, main = "PACF ")

# 6.3
cand1 <- Arima(train_ts, order = c(1, d1, 1), seasonal = c(0, D1, 1))
cand2 <- Arima(train_ts, order = c(2, d1, 1), seasonal = c(1, D1, 1))

cat("\nAIC comparison:\n")
cat("cand1 AIC:", AIC(cand1), "\n")
cat("cand2 AIC:", AIC(cand2), "\n")
cat("auto  AIC:", AIC(arima_fit), "\n")

cat("\nChosen ARIMA:\n")
print(arima_fit)


# 6.5
cat("\nResidual diagnostics in ARIMA:\n")
checkresiduals(arima_fit)

# 6.6 
fc_arima12 <- forecast(arima_fit, h = 12, level = c(80, 95))
autoplot(fc_arima12) + ggtitle("ARIMA: 12-month forecast with prediction intervals") %>% print()

# 6.7 

print(ets_fit)

# 6.8 

checkresiduals(ets_fit)

# 6.9 
fc_ets12 <- forecast(ets_fit, h = 12, level = c(80, 95))
autoplot(fc_ets12) + ggtitle("ETS: 12-month forecast with prediction intervals") %>% print()

# 6.10 
cat("\nETS vs ARIMA:\n")
cat("ARIMA RMSE:", rmse(test_ts_truth, as.numeric(arima_fc_test)), "\n")
cat("ETS   RMSE:", rmse(test_ts_truth, as.numeric(ets_fc_test)), "\n")

#-------------Random Forest and XGB

cat("\n Random Forest and XGB\n")

make_features <- function(df, max_lag = 24) {
  out <- df %>%
    mutate(
      year = year(date),
      mon  = factor(month(date))
    ) %>%
    arrange(date)
  
  for (k in 1:max_lag) out[[paste0("lag_", k)]] <- dplyr::lag(out$wind, k)
  
  out %>%
    mutate(
      rollmean_3  = (lag_1 + lag_2 + lag_3) / 3,
      rollmean_6  = (lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6) / 6,
      rollmean_12 = (lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 +
                       lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12) / 12
    ) %>%
    drop_na()
}

feat_all <- make_features(wind_df, max_lag = 24)

cutoff_date <- max(train_df$date)
train_ml <- feat_all %>% filter(date <= cutoff_date)
test_ml  <- feat_all %>% filter(date >  cutoff_date)

# RF
set.seed(42)
rf_fit <- ranger(
  dependent.variable.name = "wind",
  data = train_ml %>% select(-date),
  num.trees = 800,
  min.node.size = 5
)

rf_pred_test <- predict(rf_fit, data = test_ml %>% select(-date))$predictions

# XGB (one-hot month)
x_cols <- setdiff(names(train_ml), c("date", "wind"))
x_train <- model.matrix(~ . - 1, data = train_ml %>% select(all_of(x_cols)))
y_train <- train_ml$wind
x_test  <- model.matrix(~ . - 1, data = test_ml %>% select(all_of(x_cols)))
y_test  <- test_ml$wind

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

params <- list(
  objective = "reg:squarederror",
  eta = 0.03,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

set.seed(42)
xgb_fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1500,
  verbose = 0
)

xgb_pred_test <- predict(xgb_fit, dtest)


metrics_tbl <- tibble(
  model = c("ETS", "ARIMA", "RandomForest", "XGBoost"),
  RMSE = c(rmse(test_ts_truth, ets_fc_test),
           rmse(test_ts_truth, arima_fc_test),
           rmse(y_test, rf_pred_test),
           rmse(y_test, xgb_pred_test)),
  MAE  = c(mae(test_ts_truth, ets_fc_test),
           mae(test_ts_truth, arima_fc_test),
           mae(y_test, rf_pred_test),
           mae(y_test, xgb_pred_test)),
  MAPE = c(mape(test_ts_truth, ets_fc_test),
           mape(test_ts_truth, arima_fc_test),
           mape(y_test, rf_pred_test),
           mape(y_test, xgb_pred_test))
) %>% arrange(RMSE)

cat("\n Model Comparison \n")
print(metrics_tbl)

best_model <- metrics_tbl$model[1]
cat("\nBest model by RMSE:", best_model, "\n")


cat("\n Scenario forecasts to 2030\n")

last_date <- max(wind_df$date)
target    <- as.Date("2030-12-01")
h_to_2030 <- as.integer((year(target) - year(last_date)) * 12 + (month(target) - month(last_date)))
if (h_to_2030 < 12) h_to_2030 <- 12

future_dates <- seq(from = last_date %m+% months(1), by = "month", length.out = h_to_2030)

# Baseline forecast for best model
baseline_fc <- NULL

if (best_model == "ETS") {
  full_fit <- ets(y_ts)
  baseline_fc <- as.numeric(forecast(full_fit, h = h_to_2030)$mean)
  
} else if (best_model == "ARIMA") {
  full_fit <- auto.arima(y_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
  baseline_fc <- as.numeric(forecast(full_fit, h = h_to_2030)$mean)
  
} else {
  
  hist <- wind_df %>% arrange(date)
  
  build_one_row <- function(hist_df, next_date) {
    vals <- tail(hist_df$wind, 24)
    if (length(vals) < 24) 
    
    row <- tibble(
      date = next_date,
      year = year(next_date),
      mon  = factor(month(next_date), levels = levels(train_ml$mon))
    )
    
    for (k in 1:24) row[[paste0("lag_", k)]] <- vals[24 - k + 1]
    
    row %>%
      mutate(
        rollmean_3  = (lag_1 + lag_2 + lag_3) / 3,
        rollmean_6  = (lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6) / 6,
        rollmean_12 = (lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 +
                         lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12) / 12
      )
  }
  
  preds <- numeric(h_to_2030)
  
  for (i in seq_len(h_to_2030)) {
    nd <- future_dates[i]
    row <- build_one_row(hist, nd)
    
    if (best_model == "RandomForest") {
      p <- predict(rf_fit, data = row %>% select(-date))$predictions
    } else {
      row_x <- model.matrix(~ . - 1, data = row %>% select(all_of(x_cols)))
      p <- predict(xgb_fit, xgb.DMatrix(row_x))
    }
    
    preds[i] <- as.numeric(p)
    hist <- bind_rows(hist, tibble(date = nd, wind = preds[i]))
  }
  
  baseline_fc <- preds
}

# Scenarios based on recent growth rate ( took last 36 months)
tail_36 <- tail(wind_df$wind, 36)
g_annual <- (tail_36[length(tail_36)] / tail_36[1])^(12/36) - 1
g_annual <- max(min(g_annual, 0.30), -0.10)

optimistic_factor <- 1 + 1.5 * g_annual
slowdown_factor   <- 1 + 0.5 * g_annual

h <- 1:h_to_2030
optimistic_fc <- baseline_fc * (optimistic_factor)^(h/12)
slowdown_fc   <- baseline_fc * (slowdown_factor)^(h/12)

scenario_df <- tibble(
  date = future_dates,
  baseline   = baseline_fc,
  optimistic = optimistic_fc,
  slowdown   = slowdown_fc
)

# Plot scenarios 
plot_obs <- wind_df %>% transmute(date, series = "Observed", value = wind)
plot_scn <- scenario_df %>%
  pivot_longer(cols = c(baseline, optimistic, slowdown),
               names_to = "series", values_to = "value")

ggplot(bind_rows(plot_obs, plot_scn), aes(date, value, color = series)) +
  geom_line(linewidth = 0.8) +
  labs(title = paste0("Wind Generation Forecasts to 2030"),
       x = "Date", y = "Wind") +
  theme_minimal() %>%
  print()


dir.create("outputs", showWarnings = FALSE)

write_csv(task5_tbl,   "outputs/task5.csv")
write_csv(metrics_tbl, "outputs/models_metrics.csv")
write_csv(scenario_df, "outputs/2030_forcasts.csv")

cat("\nSaved outputs:\n")
cat(" - outputs/task5.csv\n")
cat(" - outputs/models_metrics.csv\n")
cat(" - outputs/2030_forcasts.csv\n")

