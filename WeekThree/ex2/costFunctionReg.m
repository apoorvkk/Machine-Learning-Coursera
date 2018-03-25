function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

slicedTheta = theta(2:length(theta));
h = sigmoid(X * theta);

% Calculating new J value with regularized parameters.
a = (-y .* log(h));
b = (1-y).*log(1-h);
originalCost = (1 / m) * sum(a - b);


J = originalCost + (lambda / (2*m)) * sum(slicedTheta .^ 2);


gradZero = (1/m) * sum(h-y);
slicedGrads = sum((1/m) * (h-y) .* X(:, 2:size(X, 2)), 1) + (lambda/m) * slicedTheta';
grad = horzcat(gradZero, slicedGrads)';




% =============================================================

end
