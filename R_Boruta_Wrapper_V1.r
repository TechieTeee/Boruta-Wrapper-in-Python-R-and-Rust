library(Boruta)

# Load the dataset
df <- read.csv("sales.csv")

# Separate the target variable
target <- df$revenue
df <- df[, !(colnames(df) %in% "revenue")]

# Create a Random Forest classifier object
rf <- randomForest(target ~ ., data = df, importance = TRUE)

# Create the Boruta feature selector object
boruta_selector <- Boruta(df, target, doTrace = 2)

# Fit the Boruta selector to the dataset
boruta_selector$fit(rf)

# Print the selected features
selected_features <- names(df)[boruta_selector$getSelectedAttributes()]
print(paste("Selected features:", selected_features, collapse = ", "))
