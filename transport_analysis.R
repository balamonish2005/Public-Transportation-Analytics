# Public Transportation Analytics - Ridership Prediction in R
# Monishkumar | 717825Z136 | 23CAP317 R Programming
library(ggplot2); library(corrplot); library(caret); library(ranger)

# 1. Load & explore
transport <- read.csv("transport.csv", stringsAsFactors = TRUE)
head(transport); dim(transport); names(transport); str(transport); summary(transport)

# 2. Cleaning
colSums(is.na(transport))
cat("Duplicate rows:", sum(duplicated(transport)), "\n")

# 3. Visualization
ggplot(transport, aes(x = Ridership)) + geom_histogram(bins = 40) + ggtitle("Distribution of Ridership")
num_vars <- transport[, sapply(transport, is.numeric)]
corrplot(cor(num_vars), tl.cex = 0.6)
boxplot(Ridership ~ RouteType, data = transport, main = "Ridership by RouteType")

# 4. Split
set.seed(123)
trainIndex <- createDataPartition(transport$Ridership, p = 0.8, list = FALSE)
train <- transport[trainIndex, ]; test <- transport[-trainIndex, ]

# 5. Linear regression
lm_model <- lm(Ridership ~ ., data = train); summary(lm_model)
lm_pred <- predict(lm_model, newdata = test)
lm_rmse <- sqrt(mean((lm_pred - test$Ridership)^2)); lm_r2 <- cor(lm_pred, test$Ridership)^2
cat("Linear Regression -> RMSE:", lm_rmse, " R2:", lm_r2, "\n")

# 6. Random forest
rf_model <- ranger(Ridership ~ ., data = train, num.trees = 200, importance = "impurity")
rf_pred <- predict(rf_model, data = test)$predictions
rf_rmse <- sqrt(mean((rf_pred - test$Ridership)^2)); rf_r2 <- cor(rf_pred, test$Ridership)^2
cat("Ranger Random Forest -> RMSE:", rf_rmse, " R2:", rf_r2, "\n")

# 7. Logistic regression (High / Normal demand)
transport$HighDemand <- ifelse(transport$Ridership > median(transport$Ridership), 1, 0)
train_cl <- transport[trainIndex, setdiff(names(transport), "Ridership")]
test_cl  <- transport[-trainIndex, setdiff(names(transport), "Ridership")]
glm_model <- glm(HighDemand ~ ., data = train_cl, family = binomial)
glm_probs <- predict(glm_model, newdata = test_cl, type = "response")
glm_pred_class <- ifelse(glm_probs > 0.5, 1, 0)
pred_factor <- factor(glm_pred_class, levels = c(0, 1)); ref_factor <- factor(test_cl$HighDemand, levels = c(0, 1))
print(caret::confusionMatrix(pred_factor, ref_factor, positive = "1"))

# 8. Predict a new trip
new_trip <- data.frame(Hour = 8, DayType = factor("Weekday", levels = levels(train$DayType)),
  RouteType = factor("Metro", levels = levels(train$RouteType)), Weather = factor("Clear", levels = levels(train$Weather)),
  RouteLength_km = 14.5, Fare_INR = 30, Frequency_min = 6, Temperature_C = 31, Rainfall_mm = 0,
  TrafficIndex = 75, PopDensity = 9200, NumStops = 24, SeatCapacity = 300, DelayMin = 5)
predict(lm_model, new_trip); predict(rf_model, data = new_trip)$predictions
predict(glm_model, new_trip, type = "response")
