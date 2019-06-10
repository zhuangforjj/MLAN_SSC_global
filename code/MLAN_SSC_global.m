function [result] = MLAN_SSC_global(X,c,ratio)
% X:                cell array, 1 by view_num, each array is num by d_v
% c:                number of clusters
% v:                number of views
% k:                number of adaptive neighbours
% ratio:            ratio of the labeled data in each class, 
%                   ratio = 20%, for example
% Y_L:              labeled matrix

if nargin < 5
    k = 9;
end

v = size(X,2);
num = size(X{1,1},1);
%lambda = randperm(30,1);
lambda = 10;
NITER = 30;
each_class_num = num/c;
thresh = 10^-8;
part = floor(ratio*each_class_num); % Each class have the same size of data
labeled_N = part*c;
gama = 0.2;

%% =====================   Groundtruth Generation =====================
list = sort(randperm(each_class_num,part));  % Random select the labeled data!!!
%list = [1,5,12,19,23,28]; %MSRC
%list = [7,9,12,16,26,28,31,37,39,42,44,60]; %Cal101
%list=[1,13,15,16,17,20,21,24,25,26,30,33,39,45,47,59,66,73,77,84,85,90,94,100,104,138,144,149,150,151,159,160,162,179,180,185,187,191,192,195]; %Digits
%list = [6,7,8,17,20,26,27,30,47,52,55,68,78,91,106,113,114,119,120,121,124,130,133,136,145,149,152,154,163,166,168,169,174,180,181,184,185,190,192,197]; %NUS
%list = [3,9,27,32,37,39,41,48,52,56,57,60,64];%YaleB
%list =[1,3,14,15,16,19,26,28,32,42,43,49,60,66,70,77,82,92,95,100,107,112,113,122,127,131,137,138,146,147,152,156,164,165,169,175,177,185,192,193]; %UCI
List = [];
for c = 1:c
    List = [List list+(c-1)*each_class_num];
end
List_ = setdiff(1:1:num,List); % the No. of unlabeled data
    
samp_label = zeros(num,c); % column vector
for c = 1:c
    samp_label((c-1)*each_class_num+(1:each_class_num),c) = ones(each_class_num,1);
end

groundtruth = zeros(num,c);
groundtruth(1:labeled_N,:) = samp_label(List,:);
groundtruth((labeled_N+1):num,:) = samp_label(List_,:);
Y_L = groundtruth(1:labeled_N,:);

%% =====================   Normalization =====================
for i = 1 :v
    for  j = 1:num
        X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
    end
end

%全局结构生成
glo_fea = [];    
for i = 1:v
    fea_v = X{i}'; 
    glo_fea = [glo_fea;fea_v];    
    distX_initial(:,:,i) =  L2_distance_1( X{i}',X{i}' ) ; 
end
    distX_initial(:,:,v+1) =  L2_distance_1( glo_fea,glo_fea ) ; 
    X{v+1} = glo_fea';
   
%样本按照标签顺序排列
for i = 1 :v+1
    temp = X{i};
    X{i}(1:labeled_N,:) =  temp(List,:);
    X{i}((labeled_N+1):num,:) = temp(List_,:); 
end

%% =====================  Initialization =====================

%initialize weighted_distX
SUM = zeros(num);
for i = 1:v+1
    if i == v+1
        distX_initial(:,:,v+1) =  gama*L2_distance_1( glo_fea,glo_fea ) ; 
    else
        distX_initial(:,:,i) =  L2_distance_1( X{i}',X{i}' ) ;                  %initialize X
    end
    SUM = SUM + distX_initial(:,:,i);
end
distX = (1/(v+1))*SUM;
[distXs, idx] = sort(distX,2);

%initialize S
S = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distXs(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);               %initialize S
end;
alpha = mean(rr);

% initialize F
S = (S+S')/2;                                                         % initialize F
D = diag(sum(S));
L = D - S;
[F, temp, evs]=eig1(L, c, 0);

if sum(evs(1:c+1)) < 0.00000000001
    error('The original graph has more than %d connected component', c);
end;

%% =====================  updating =====================
for iter = 1:NITER
    % update weighted_distX
    SUM = zeros(num,num);
    for i = 1 : v+1
        if iter ==1
            distX_updated = distX_initial;
        end
        if i == v+1
           Wv(i) = 0.5/sqrt(sum(sum( (distX_updated(:,:,i)/(gama+eps)).*S)));  
        else
           Wv(i) = 0.5/sqrt(sum(sum( distX_updated(:,:,i).*S)));                % update X
        end
        distX_updated(:,:,i) = Wv(i)*distX_updated(:,:,i) ;
        SUM = SUM + distX_updated(:,:,i);
    end
    distX = SUM;
    %[distXs, idx] = sort(distX,2);  %%自己加的
    
    %update S
    distf = L2_distance_1(F',F');
    S = zeros(num);
    for i=1:num                                                         %update A
        idxa0 = idx(i,2:k+1);
        dfi = distf(i,idxa0);
        dxi = distX(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*alpha);
        S(i,idxa0) = EProjSimplex_new(ad);
    end;
    
    %update F
    S = (S+S')/2;                                                        %update F
    D = diag(sum(S));
    L = D-S;
    L_uu = L((part*c+1):end, (part*c+1):end);
    L_ul = L((part*c+1):end, 1:part*c);
    F_u = (-1)*inv(L_uu)*L_ul*Y_L;
    F = [Y_L;F_u];
    
    
    Sum = 0;
    for i = 1:v+1
        Sum = Sum + Wv(i)*trace(distX_updated(:,:,i)'*S);
    end
    obj(iter) = Sum  + alpha*norm(S,2)^2
    if iter>2 && ( obj(iter-1)-obj(iter) )/obj(iter-1) < thresh
        break;
    end
  sprintf('iter = %d',iter)
end

%% =====================  result =====================

F_discrete = zeros(num,c);
[max_value,max_ind] = max(F,[],2);
for i = 1: num
    F_discrete(i,max_ind(i)) = 1;
end
cnt = 0;
for n = 1:num
    if F_discrete(n,:) == groundtruth(n,:);
        cnt = cnt+1;
    end
end
result= (cnt-part*c)/(num-part*c);
