library(stats)
library(MASS)
library(lpSolve)
library(glmnet)
library(ROI)





Lasso_UP <-function(X,y,lambda,method = "QP",penidx=NULL,A=NULL,b=NULL) {
    X <- as.matrix(X)
    n <- dim(X)[1]
    p <- dim(X)[2]
    y <- as.matrix(y)
    dim(y) <- c(n, 1)
    
    mean.y <- mean(y) 
    young <- which(y<mean.y)
    old <- which(y>mean.y)
    
    A1 <-  matrix(rep(1/length(young),length(young)),nrow=1)%*%as.matrix(X[young,])
    A2 <- matrix(rep(1/length(old),length(old)),nrow=1)%*%as.matrix(X[old,])
    Aeq <- rbind(A1,A2)
    
    
    sum.young <- as.matrix(mean(y[young]))
    sum.old <- as.matrix(mean(y[old]))
    beq <- c(sum.young,sum.old)
    
  
    
    
    if (is.null(penidx)) {
      penidx <- matrix(TRUE, p, 1)
    }
    dim(penidx) <- c(p, 1)
    
    if (is.null(A)) {
      A <- matrix(0, 0, p)
      b <- rep(0, 0)
    }
    
    if (is.null(Aeq)) {
      Aeq <- matrix(0, 0, p)
      beq <- rep(0, 0)
    }
    
    m1 <- dim(Aeq)[1]
    m2 <- dim(A)[1]
    
    if (is.null(lambda)) {
      stop(cat("Please, enter a value for the penalty parameter lambda."))
    }
    
    ## no constraints
    if (m1 == 0 && m2 == 0) {
      # no penalization
      if (abs(lambda) < 1e-16) {
        wt <- matrix(1, n, 1)
        dim(wt) <- c(n, 1)
        Xwt <- X * as.numeric(sqrt(wt))
        ywt <- as.numeric(sqrt(wt)) * y
        
        betahat <- matrix(stats::lm(ywt ~ Xwt - 1)$coef, p, 1)
        dual_eq <- rep(0, 0)
        dual_neq <- rep(0, 0)
      } else {
        # with penalization
        if (method == "CD") {
          glmnet_res <-
            glmnet::glmnet(
              Xwt,
              ywt,
              alpha = 1,
              lambda = lambda,
              penalty.factor = penidx,
              maxit = 1000,
              intercept = FALSE,
              standardize = TRUE
            )
          
          betahat <- matrix(glmnet_res$beta, p, 1)
          dual_eq <- rep(0, 0)
          dual_neq <- rep(0, 0)
        } else if (method == "QP") {
          wt <- matrix(1, n, 1)
          dim(wt) <- c(n, 1)
          Xwt <- X * as.numeric(sqrt(wt))
          ywt <- as.numeric(sqrt(wt)) * y
          
          # quadratic coefficient
          H <- t(Xwt) %*% Xwt
          H <- rbind(cbind(H, -H), cbind(-H, H))
          
          # linear coefficient
          f <- -t(Xwt) %*% ywt
          f <- rbind(f, -f) + lambda * rbind(penidx, penidx)
          
          # optimizer
          x <- ROI::OP(ROI::Q_objective(H, L = t(f)))
          opt <- ROI::ROI_solve(x, solver = "qpoases")
          opt_sol <- opt$message$primal_solution
          
          # estimators
          betahat <-
            matrix(opt_sol[1:p] - opt_sol[(p + 1):length(opt_sol)], p, 1)
          dual_eq <- rep(0, 0)
          dual_neq <- rep(0, 0)
        }
      }
    } else {
      ## with constraints
      
      if (method == "CD") {
        warning("The CD method does not work with constraints. The solution is generated with QP.")
      }
      
      wt <- matrix(1, n, 1)
      dim(wt) <- c(n, 1)
      Xwt <- X * as.numeric(sqrt(wt))
      ywt <- as.numeric(sqrt(wt)) * y
      
      # quadratic coefficient
      H <- t(Xwt) %*% Xwt
      H <- rbind(cbind(H, -H), cbind(-H, H))
      
      # linear coefficient
      f <- -t(Xwt) %*% ywt
      f <- rbind(f, -f) + lambda * rbind(penidx, penidx)
      
      # constraints
      Amatrix <- rbind(cbind(Aeq, -Aeq), cbind(A, -A))
      bvector <- c(beq, b)
      
      # optimizer
      x <-
        ROI::OP(
          ROI::Q_objective(H, L = t(f)),
          ROI::L_constraint(
            L = Amatrix,
            dir = c(rep("==", m1), rep("<=", m2)),
            rhs = bvector
          )
        )
      opt <- ROI::ROI_solve(x, solver = "qpoases")
      opt_sol <- opt$message$primal_solution
      
      # estimators
      betahat <-
        matrix(opt_sol[1:p] - opt_sol[(p + 1):length(opt_sol)], p, 1)
      duals <- opt$message$dual_solution[-(1:(2 * p))]
      
      if (m1 != 0) {
        dual_eq <- -matrix(duals[1:m1], m1, 1)
      } else {
        dual_eq <- rep(0, 0)
      }
      if (m2 != 0) {
        dual_neq <- -matrix(duals[(m1 + 1):(m1 + m2)], m2, 1)
      } else {
        dual_neq <- rep(0, 0)
      }
    }
    
    predicted <- X%*%betahat
    
    return(list(
      "betahat" = betahat,
      "dual_eq" = dual_eq,
      "dual_neq" = dual_neq,
      "predict" = predicted
    ))
  }
