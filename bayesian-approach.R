# Clear the workspace
rm(list = ls())
install.packages("lmtest")
install.packages("car")
install.packages("GGally")


# Load necessary libraries
library(reshape2)
library(rstudioapi)
library(rstan)
library(ggplot2)
library(dplyr)
library(statsr)
library(BAS)
library(psych)
library(rstanarm)
library(bayesplot)
library(lmtest) # for Durbin-Watson test
library(car)    # for VIF and added variable plots

# Stan and Parallel Options
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Read Data
train <- read.csv("C:/Users/swath/OneDrive/Desktop/baysian statistics/kc_house_data.csv/kc_house_data.csv")
train$date <- as.Date(substr(train$date, 1, 8), format="%Y%m%d")

# Data Transformation
train$ym <- paste0(format(train$date, "%Y"), "-", format(train$date, "%m"))
train <- subset(train, select = -date)
train$ym <- as.factor(train$ym)
train$ym <- relevel(train$ym, "2014-05")

# Data Preprocessing
# Handling missing values - remove rows with any missing values
train <- na.omit(train)

# Checking for data entry errors - ensuring bedrooms are in a plausible range
train <- train[train$bedrooms >= 0 & train$bedrooms <= 10, ]

# Log Transformations and New Variables
train$log_price <- log(train$price)
train$log_sqft_lot <- log(train$sqft_lot)
train$log_sqft_basement <- log(train$sqft_basement + 1)
train$log_sqft_above <- log(train$sqft_above + 1)
train$basement <- ifelse(train$sqft_basement == 0, 0, 1)
train$renovated <- ifelse(train$yr_renovated == 0, 0, 1)


# Let's generate a histogram using the provided ggplot2 code syntax.
# However, since we do not have the actual dataset 'train', we will create a simulated dataset to demonstrate.

# Assuming 'log_price' is normally distributed for the purpose of this example.
# Normally, you would use the actual data from your 'train' dataframe.

# Load the ggplot2 library
library(ggplot2)

# Generate a simulated 'log_price' column with normally distributed data
set.seed(123)  # for reproducibility
log_price <- rnorm(1000, mean = 10, sd = 0.5)  # mean and sd are arbitrary

library(ggplot2)

ggplot(train, aes(x = bedrooms, y = log_price)) +
  geom_point(aes(color = factor(bedrooms)), alpha = 0.6, size = 3) +  # Color code by number of bedrooms
  geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "dashed") +  # Add a linear trend line
  scale_color_viridis_d() +  # Use a color scale that is visually appealing and accessible
  labs(
    title = "Log Price vs. Number of Bedrooms",
    x = "Number of Bedrooms",
    y = "Log-transformed Price",
    color = "Bedrooms"
  ) +
  theme_minimal() +  # Use a minimal theme for a clean look
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),  # Center and bold the plot title
    axis.title = element_text(size = 14, face = "bold"),  # Bold axis titles
    legend.position = "bottom"  # Move the legend to the bottom
  ) +
  guides(color = guide_legend(title = "Number of Bedrooms"))  # Add a legend title



# Splitting the data into training and testing sets
set.seed(123)  # for reproducibility
train_indices <- sample(1:nrow(train), size = 0.8 * nrow(train))
train_data <- train[train_indices, ]
test_data <- train[-train_indices, ]

# Filtering and Demeaning
train_data <- subset(train_data, !train_data$bedrooms %in% c(11, 33))
train_data <- transform(train_data, demean_bedrooms = bedrooms - mean(bedrooms),
                        demean_bathrooms = bathrooms - mean(bathrooms),
                        demean_floors = floors - mean(floors),
                        demean_view = view - mean(view),
                        demean_condition = condition - mean(condition),
                        demean_grade = grade - mean(grade),
                        demean_log_sqft_above = log_sqft_above - mean(log_sqft_above),
                        demean_log_sqft_basement = log_sqft_basement - mean(log_sqft_basement),
                        demean_log_sqft_lot = log_sqft_lot - mean(log_sqft_lot),
                        demean_yr_built = yr_built - mean(yr_built))

# Model Data
model_data <- model.matrix(log_price ~ demean_bedrooms + demean_bathrooms + demean_floors + waterfront + 
                             demean_view + demean_condition + demean_grade + demean_yr_built + yr_renovated + 
                             demean_log_sqft_above + demean_log_sqft_basement + demean_log_sqft_lot + ym + 
                             renovated + basement, train_data)

# Zipcode Data
mn <- length(unique(train_data$zipcode))
zip <- data.frame(zipcode = unique(train_data$zipcode), index_zipcode = 1:mn)
train_data <- merge(train_data, zip, by = "zipcode")

# Linear Regression Model
linear_model <- lm(log_price ~ ., data = train_data)

# Summary of Linear Model
summary_linear <- summary(linear_model)

# Model Diagnostics
# Plotting residuals to check for homoscedasticity and linearity

par(mfrow = c(2, 2))
plot(linear_model)


# Checking for normality of residuals
qqnorm(linear_model$residuals)
qqline(linear_model$residuals)

# Checking for independence (autocorrelation) of residuals
dwtest(linear_model)

# Identify potential outliers and influential points
cooks.distance <- cooks.distance(linear_model)
influential_points <- which(cooks.distance > (4/length(cooks.distance)))

# Check for linear combinations and redundant variables
alias(linear_model)


linear_model <- lm(log_price ~ zipcode + id + waterfront + sqft_above + 
                     yr_renovated + lat + long + sqft_living15 + sqft_lot15 + 
                     ym + basement + renovated + demean_bedrooms + demean_bathrooms + 
                     demean_floors + demean_view + demean_condition + demean_grade + 
                     demean_log_sqft_above + demean_log_sqft_basement + 
                     demean_log_sqft_lot + demean_yr_built + index_zipcode, 
                   data = train_data)





# After addressing the issues, try calculating VIF again
vif(linear_model)

# Identify influential points using Cook's distance
cooks_distance <- cooks.distance(linear_model)

# Set threshold for influential points (commonly used threshold is 4/n)
threshold <- 4 / nrow(train_data)

# Find the indices of influential points
influential_points <- which(cooks_distance > threshold)

# Remove influential points from the data
train_data_clean <- train_data[-influential_points, ]

library(ggplot2)
library(GGally)

# Select relevant numerical variables
numerical_vars <- c("log_price", "demean_bedrooms", "demean_bathrooms", "sqft_above", "yr_renovated")

# Create scatterplot matrix
ggpairs(train_data_clean, columns = numerical_vars, title = "Pairwise Scatterplot Matrix")



library(reshape2)

# Calculate correlation matrix
corr_matrix <- cor(train_data_clean[, numerical_vars])

# Melt the correlation matrix
corr_matrix_melted <- melt(corr_matrix)

# Create a heatmap
ggplot(corr_matrix_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Correlation Heatmap")


library(ggplot2)

train_data_clean$renovated <- factor(train_data_clean$renovated)

ggplot(train_data_clean, aes(x = renovated, y = log_price, fill = renovated)) +
  geom_boxplot() +
  scale_fill_manual(values = c("0" = "red", "1" = "green")) +
  labs(title = "Boxplot of Log Price by Renovated Status", x = "Renovated", y = "Log Price")



library(ggplot2)

# Create density plots for numeric variables
ggplot(train_data_clean, aes(x = log_price, fill = basement)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Log Price by Basement Status", x = "Log Price") +
  facet_grid(. ~ basement)

library(ggplot2)

# Create a bar chart for zipcodes
ggplot(train_data_clean, aes(x = as.factor(zipcode))) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Distribution of Zipcodes", x = "Zipcode", y = "Count")




set.seed(123)  # for reproducibility
train_indices <- sample(1:nrow(train_data_clean), size = 0.8 * nrow(train_data_clean))
train_data_split <- train_data_clean[train_indices, ]
test_data_split <- train_data_clean[-train_indices, ]


# Refit the linear regression model without influential points
linear_model_clean <- lm(log_price ~ ., data = train_data_split)


# Summary of the refitted model
summary_linear_clean <- summary(linear_model_clean)

# Diagnostic plots for the refitted model

par(mfrow = c(2, 2))
plot(linear_model_clean)






# Bayesian Linear Regression Model using the cleaned dataset
bayesian_linear_model <- stan_glm(log_price ~ ., data = train_data_split, 
                                  prior = normal(0, 2.5), 
                                  chains = 2, iter = 1200, 
                                  warmup=20, )  # Closing parenthesis added here

# Summary of Bayesian Linear Model
summary_bayesian <- summary(bayesian_linear_model)


# Diagnostic Plots for Bayesian Linear Model
bayesian_preds <- posterior_predict(bayesian_linear_model, newdata = test_data_split)
png("bayesian_linear_model_predicted_vs_observed.png")
plot(test_data$log_price, apply(bayesian_preds, 2, mean))
abline(0, 1, col = "red")
dev.off()


# Set color scheme for plots
color_scheme_set("brightblue")

# Extract posterior samples
posterior_samples <- as.matrix(bayesian_linear_model)

# R-hat plot
mcmc_rhat(posterior_samples)

# ESS plot
mcmc_ess_bulk(posterior_samples)
mcmc_ess_tail(posterior_samples)

# Traceplot for visualizing chains (Optional, for further diagnostics)
mcmc_trace(posterior_samples)



# Hierarchical Linear Regression Model (Stan Model)
stan_code_intercept <- "
data {
  int<lower=1> K;
  int<lower=1> J;
  int<lower=1> N;
  matrix[N, K] x;
  vector[N] y;
  int zipcode[N];
}
parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_a;
  real mu;
  vector[J] a;
  vector<lower=0>[K] beta;
  real mu_a;
}
transformed parameters {
  real mu_adj;
  vector[J] a_adj;
  mu_adj = mu + mean(a);
  a_adj = a - mean(a);
}
model {
  a ~ normal(mu_a, sigma_a);
  mu_a ~ normal(0, 100);
  y ~ normal(mu_adj + a_adj[zipcode] + x * beta, sigma);
}
generated quantities {
  vector[N] y_hat;
  y_hat = mu_adj + a_adj[zipcode] + x * beta;
}
"
# Ensure that model_data is created based on the split training data
model_data <- model.matrix(log_price ~ ., data = train_data_split)

# Now create x, y, and zipcode based on the split training data
x <- model_data[, -1]  # Remove intercept column
y <- train_data_split$log_price
zipcode <- train_data_split$index_zipcode

# Ensure N is consistent with the number of observations in train_data_split
N <- nrow(train_data_split)

# Stan data list
stan_data <- list(K = ncol(x), J = length(unique(zipcode)), N = N, x = x, y = y, zipcode = zipcode)

# Run Stan Model
fit <- stan(model_code = stan_code_intercept, data = stan_data, iter = 4000, chains = 2, 
            control = list(max_treedepth = 20), verbose = TRUE)


# Checking the fit object
print(fit)  # Should provide a summary of the Stan fit object

# Save the trace plots to files
png("traceplot_beta.png")
stan_trace(fit, pars = "beta")
stan_trace(fit, pars = "a_adj")
dev.off()

png("traceplot_sigma.png")
stan_trace(fit, pars = "sigma")
dev.off()

png("pairs_plot.png")
stan_pairs(fit, pars = c("beta", "sigma", "mu_a", "a", "sigma_a"))
dev.off()

# Model Comparison and Evaluation

# Linear model predictions on test data
linear_preds <- predict(linear_model, newdata = test_data_split)

# Bayesian linear model predictions on test data
bayesian_preds_mean <- apply(bayesian_preds, 2, mean)

# Hierarchical model predictions on test data
# Note: Adjust if the structure of 'fit' is different
hierarchical_preds <- extract(fit)$y_hat
hierarchical_preds_mean <- apply(hierarchical_preds, 2, mean)

# Function to calculate RMSE
calculate_rmse <- function(predictions, actual) {
  sqrt(mean((predictions - actual)^2))
}

# Calculate RMSE for each model
linear_rmse <- calculate_rmse(linear_preds, test_data_split$log_price)
bayesian_rmse <- calculate_rmse(bayesian_preds_mean, test_data_split$log_price)
hierarchical_rmse <- calculate_rmse(hierarchical_preds_mean, test_data_split$log_price)

# Comparison Table
comparison_results <- data.frame(
  Model = c("Linear", "Bayesian Linear", "Hierarchical"),
  RMSE = c(linear_rmse, bayesian_rmse, hierarchical_rmse)
)

print(comparison_results)







