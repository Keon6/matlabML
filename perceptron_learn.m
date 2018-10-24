function [w, iterations] = perceptron_learn(data_in)
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
    input_dim = size(data_in);
    % Number of Datapoints = input_dim(1) = number of rows
    % Number of Features excluding Class = input_dim(2)-1 = number of columns - 1
    
    % Initialize outputs
    w = zeros(1, input_dim(2)); % Initialize w(0) = 0
    iterations = 0;
    
    % Separate X and Y
    X = data_in(:,1:input_dim(2)-1);
    Y = data_in(:,input_dim(2));
    % Algorithm
    exit = false;
    while exit == false
        iterations = iterations + 1;
        H = (sign(w(2:input_dim(2))*X' + w(1)))'; % Classification with model w(t)
        H(H == 0) = 1; % If x lies on w(t), classify as +1
        if isequal(Y, H) % If every datapoint is correctly classified
            exit = true;
        else % If wrongly classified
            misclassified = find((Y == H) == 0); % Find all misclassified indices
            y = Y(misclassified(1));
            x = X(misclassified(1), :);
            w(2:input_dim(2)) = w(2:input_dim(2)) + y*x; % w(t+1) = w(t) + y(t)x(t)
        end
    end
end

