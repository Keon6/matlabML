function [w, e_in] = logistic_reg(X, y, w_init, max_its, eta)
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)

    % Initial preprocessing  - adding column of 1s to X, threshold for algo stop and so on...
    N = size(X); d = N(2); N = N(1); 
    X = [ones(N,1) X];
    w = w_init;
    
    % Set threshold for stopping alo, Initialize number of iterations and gradient
    convergence_thresholds = (10^-3)*ones(d+1, 1); 
    it = 0; 
    g_t = inf(d+1, 1);
    
    % Run GD:
    i = 1:N; %each data point
    while and(it<=max_its, sum(g_t<=convergence_thresholds)<d+1)
    %while sum(g_t<=convergence_thresholds)<d+1
        g_t = transpose((-1/N)*sum(y(i).*X(i,:)./(1+exp(y(i).*X(i,:)*w)))); % gradient
        v_t = - g_t/norm(g_t);
        w = w + eta*v_t;  
        it = it + 1;
    end
    % Find error after GD stops
    disp(it)
    e_in = (1/N)*sum(log(1+exp(-y(i).*X(i,:)*w)));
end

