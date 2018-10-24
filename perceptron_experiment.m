function [num_iters, bounds] = perceptron_experiment (N, d, num_samples)
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bounds theoretical upper bound for number of iterations
%      (both the outputs should be num_samples long)
    % initialize number of iterations & bounds at 0
    num_iters = zeros(num_samples, 1); 
    bounds = zeros(num_samples, 1); % From LFD 1.3 e), t<= R^2||w_star||^2/rho^2
    % R = max_1<=n<=N {||x_n||}
    % rho = min_1<=n<=N {y_n(x_n * w_star)}
    for i=1:num_samples
        % Randomly generate d+1 Dimensional weight vector - uniform from (0, 1)
        w_star = [0 rand(1, d)]; % (1 x d+1)
        % Randomly generate (N x d) training data - uniform from (-1, 1)
        X = -1 + 2*rand(N, d); %N rows(datapoints) x d columns(features/attributes)
        w_star_X = w_star(2:d+1)*X' + w_star(1); %(1 x N)
        Y = sign(w_star_X); %(1 x N) 
        Y(Y == 0) = 1; % If x lies on w_star, then classify it +1
        %D = [X Y'];
        rho = min(w_star_X.*Y);
        R = max(sqrt(sum(X.^2,2))); %sqrt(sum(X.^2,2))=norms of row vectors
        bounds(i) = floor(((R*norm(w_star))^2)/(rho^2)); %Initialize bound as theoratical bound
        % PLA with w(0) = 0 on the training set
        [w, num_iters(i)] = perceptron_learn([X Y']);
    end
    %bounds_minus_ni = bounds_minus_ni - num_iters; % bounds
end

