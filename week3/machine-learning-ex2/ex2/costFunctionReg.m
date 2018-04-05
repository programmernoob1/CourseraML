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
J=-1*(y'*log(sigmoid(X*theta))+(1-y')*log(1-sigmoid(X*theta)))/m;
dummy=theta;
dummy(1)=0;
J=J+(sum(dummy.^2))*lambda/(2*m);

grad=(sigmoid(X*theta)-y)'*X/m;
grad=grad';
change=zeros(size(theta));
change=lambda*theta/m;
change(1)=0;
grad=grad+change;





% =============================================================

end
