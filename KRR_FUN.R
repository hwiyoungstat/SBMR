
library(quadprog)

gaussian_kernel <- function(X1, X2, sigma) {
  sq_dist <- as.matrix(dist(rbind(X1, X2)))^2
  K <- matrix(sq_dist[1:nrow(X1), -seq_len(nrow(X1))], nrow = nrow(X1))
  K <- exp(-K / (2 * sigma^2))
  return(K)
}



kernel_ridge_regression <- function(X, y, lambda, sigma) {
  # Compute the Gaussian kernel matrix
  K <- gaussian_kernel(X, X, sigma)
  
  # Number of training examples
  n <- nrow(X)
  
  # Solve for alpha using the ridge regression formula
  # (K + lambda * I)^-1 * y
  I <- diag(n)
  alpha <- solve(K + lambda * I) %*% y
  
  # Return the model parameters and the training data
  return(list(alpha = alpha, X_train = X, sigma = sigma))
}


predict_krr <- function(model, X_new) {
  # Compute the kernel matrix between new data points and training data
  K_new <- gaussian_kernel(X_new, model$X_train, model$sigma)
  
  # Predict the target values for the new data
  y_pred <- K_new %*% model$alpha
  
  return(y_pred)
}






kernel_ridge_regression_eq <- function(X, y, lambda, sigma) {
  # Compute the Gaussian kernel matrix for the training data
  K <- gaussian_kernel(X, X, sigma)
  
  # Setup the quadratic programming problem
  Dmat <- (K + lambda * diag(nrow(K)))
  dvec <- y
  
  
  
  my <- mean(y)
  young <- which(y<my)
  old <- which(y>my)
  
  K.young <-gaussian_kernel(matrix(X[young,],nrow=length(young)),X,sigma)
  K.old <-gaussian_kernel(matrix(X[old,],nrow=length(old)),X,sigma)
  
  one <- matrix(rep(1,length(y)),nrow=1)
  
  one.young <- matrix(rep(1,length(y[young])),nrow=1)
  one.old <- matrix(rep(1,length(y[old])),nrow=1)

  A1 <- one.young%*%K.young
  A2 <- one.old%*%K.old
  
  Amat <- rbind(A1,A2)
  
  y.young <- sum(y[young])
  y.old <- sum(y[old])
  bvec <- rbind(y.young,y.old)
  
  
  
  # Solve the quadratic programming problem
  sol <- solve.QP(Dmat, dvec, t(Amat), bvec, meq = nrow(Amat))
  alpha <- sol$solution
  
  est <- K%*%alpha
  
  return(list(alpha = alpha, X_train = X, y_train = y, lambda = lambda, sigma = sigma, predict = est))
}

