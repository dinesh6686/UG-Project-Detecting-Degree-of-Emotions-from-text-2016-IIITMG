clear all
clc

addpath(genpath('./Data'));
addpath(genpath('./EmoDetect'));

load S.mat
% ground truth emotion-sentence matrix, contains S_hat_train, S_hat_test
% each row (from up to dwon) corresponds to one emotion in the order of surprise, anger, fear, joy and sad

load D_hat.mat
% words-emotion matrix, contains emotion for each words in the dictionary

load X.mat
% words-sentence matrix, contains X_train, X_test, 

load W.mat 
% W is correlation matrix between training sentences
% it can be computed by
% W = 1 - pdist2(X_train', X_train','cosine'); 
% W(isnan(W))=eps;

load W_test.mat;
% W_test is correlation matrix between training and testing sentences
% it can be computed by
% W_test = 1 - pdist2(X_test',X_train','cosine'); 
% W_tr(isnan(W_tr))=eps;

%% parameter setting
[m,n] = size(X_train);              % m: words, n: sentences
nn = size(S_hat_test,2);            % num of sentences in test data
k = 5;                              % num of emotions
K = size(D_hat,1);                  % num words in dict

%% Initialization
D = rand(m,k);
S = rand(k,n);
A = S;
B = D;
maxiter = 100;

%% emotion detection
[D,S_train] = emodetect(X_train, W, D, S, B, A, D_hat, S_hat_train, K, maxiter);

%% compute S_test
S_test = D'*X_test + 0.001*(W_test*pinv(S_train))';

%% select number of emotions in each sentence
S_test = emoselect(S_test,5,k); % 1 mean 2 med 3 auto 4 normalize & round 5 max

%% compute precision recall f-score
p_test = (sum(S_test & S_hat_test,2)) ./ (sum(S_test,2));
r_test = (sum(S_test & S_hat_test,2)) ./ (sum(S_hat_test,2));
f_test = 2*(p_test.*r_test)./(p_test+r_test);
fprintf(' precision : %7.1e recall : %7.1e f-score : %7.1e\n', mean(p_test), mean(r_test), mean(f_test));