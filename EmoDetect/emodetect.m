function [D,S] = emodetec(X_train, W, D, S, B, A, D_hat, S_hat, K, maxiter)

epsilon = 0.00001;
lambda_w = 0.001;
lambda_q = 0.01;
lambda_d = 1;
lambda_s = 1;
lambda_a = 1;
lambda_b = 1;

% get size
[m,n] = size(X_train); [~,r] = size(D);

% get L 
gamma = 0.5;
L = zeros(n,n);
L(W>=gamma) = 1;

% Q: emotion correlation matrix
foo = 0.01;                             
Q = [1,foo,0,0,foo;
    foo,1,foo,0,foo;
    0,foo,1,0,foo;
    0,0,0,1,0;
    foo,foo,foo,0,1,];                    

% fill D_hat with zeros for words with no ground truth emotion
D_hat = [D_hat;zeros(size(D(K+1:end,:)))]; 

% Main loop
iter = 0;
errtol = 1e-3;

while iter <= maxiter 
    
    Dold = D;
    Sold = S;
    
    % Update of D
    D1 = D.*((X_train*S'+lambda_q*B*Q+epsilon+lambda_d*D_hat + lambda_b*B)./(X_train*X_train'*D+lambda_q*(B*B')*D+epsilon+lambda_d*D + lambda_b*D));
    D2 = D.*((X_train*S'+lambda_q*B*Q+epsilon + lambda_b*B)./(X_train*X_train'*D+lambda_q*(B*B')*D+epsilon+lambda_b*D));
    D = [D1(1:K,:);D2(K+1:end,:)];
    
    % Update of S
    S = S.*((D'*X_train+lambda_w*A*(L.*W)+epsilon+lambda_s*S_hat + lambda_a*A)./(S+lambda_w*A*(L.*(A'*S))+epsilon+lambda_s*S + lambda_a*S ));
    
    % Update of A(S)
    A = A.*((lambda_w*S*(L.*W) + lambda_a*S + epsilon)./( lambda_w*S*(L.*(S'*A)) + lambda_a*A + epsilon));
    
    % Update of B(D)
    B = B.*((lambda_q*D*Q + lambda_b*D + epsilon)./( lambda_q*D*D'*B + lambda_b*B + epsilon));
    iter = iter+1;
    
    err = max(norm(Sold - S)/norm(S), norm(Dold - D)/norm(D));
    
    fprintf(' iter: %2d relative error: %7.1e\n', iter, err);
    if err < errtol
        break;
    end
end
end