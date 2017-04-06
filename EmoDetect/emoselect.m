function S = emoselect(S,verbose,k)
% select number of emotions per sentence in S_test
% S: emotion-by-sentence matrix
% m: maximum of emotions to select
% k: number of emotions
num_s = size(S,2);
nn = size(S,2);
switch verbose
    case 1  % mean
        for j = 1 : num_s
            S_j= S(:,j);
            mean_j = mean(S_j);
            S_j(S_j >= mean_j) = 1;
            S_j(S_j < mean_j) = 0;
            S(:,j) = S_j;
        end
    case 2  % med
        for j = 1 : num_s
            S_j= S(:,j);
            med_j = median(S_j);
            S_j(S_j > med_j) = 1;
            S_j(S_j <= med_j) = 0;
            S(:,j) = S_j;
        end
    case 3  % flexible
        for j = 1 : num_s
            S_j= S(:,j);
            [max_j,id] = max(S_j);
            for i = 1:k
                if max_j-S_j(i) < 0.1 & S_j(i) > 0.09;
                    S_j(i) = 1;
                else
                    S_j(i) = 0;
                end
            end
            S(:,j) = S_j;
        end
        
    case 4   % normalzie & round
        col_sum = sum(S,1);
        col_sum = col_sum(ones(k,1),:);
        S = S./col_sum;
        S = round(S);
        
    case 5 % select max
        [val,id] = max(S,[],1);
        for  i = 1: nn
            if val(i) ~= 0
                S(id(i),i) = 1;
            end
        end
        S(S~=1)=0;
        
    case 6 % min-max scaling
        mn = repmat(min(S')',1,size(S,2));
        mx = repmat(max(S')',1,size(S,2));
        S = (S-mn)./(mx-mn);
        S = round(S);

end