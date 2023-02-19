library(Boruta)

# Define a decorator function to time function execution
timeit <- function(f) {
  function(...) {
    start_time <- Sys.time()
    result <- f(...)
    end_time <- Sys.time()
    execution_time <- end_time - start_time
    cat("Execution time:", execution_time, "\n")
    return(result)
  }
}

# Define a Boruta feature selector class
BorutaSelector <- setClass(
  "BorutaSelector",
  slots = c(
    df = "data.frame",
    target = "numeric",
    boruta_selector = "Boruta",
    selected_features = "character"
  ),
  prototype = list(
    df = data.frame(),
    target = numeric(),
    boruta_selector = NULL,
    selected_features = character()
  )
)

# Define a method to fit the Boruta feature selector to the dataset
BorutaSelector$methods(
  fit = function(object) {
    rf <- randomForest(object@target ~ ., data = object@df, importance = TRUE)
    object@boruta_selector <- Boruta(object@df, object@target, doTrace = 2)
    object@boruta_selector$fit(rf)
    object@selected_features <- names(object@df)[object@boruta_selector$getSelectedAttributes()]
    return(object)
  }
)

# Define a method to print the selected features
BorutaSelector$methods(
  print = function(object) {
    message("Selected features: ", paste(object@selected_features, collapse = ", "))
  }
)

# Load the dataset
df <- read.csv("sales.csv")

# Create a Boruta feature selector object and fit it to the dataset
boruta_selector <- BorutaSelector(df = df[, !(colnames(df) %in% "revenue")], target = df$revenue)
boruta_selector <- timeit(boruta_selector@fit)()(boruta_selector)

# Print the selected features
print(boruta_selector)
