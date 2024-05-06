close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);
end
tic;
dataset_name = {'COCO'};%MIRFLICKR NUSWIDE21
param.top_K = 2000;
for dbi = 1     :length(dataset_name)
    dataset_name = dataset_name{dbi}; param.dataset_name = dataset_name;
    turn = 2;          nbits = [16];
    %% load dataset
load(['./datasets/',dataset_name,'_deep.mat'])
if strcmp(dataset_name, 'MIRFLICKR')
alpha = [0.7];
pace = [0.05];
lambda = [1000];
muta = 100;
theta = [10000];
query_size = 2000;
train_size = 10000;
% alpha = [0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
%    lambda = [0,0.1,1,10,100, 1000, 10000,100000];
% pace = [0.00005 0.0001 0.0005 0.001  0.005 0.01 0.05 0.1 0.5 1];
%    muta = [0,0.1,1,10,100, 1000, 10000,100000];
%                  theta = [0,0.01,0.1,1,10,100, 1000,10000, 100000];

elseif strcmp(dataset_name, 'NUSWIDE21')
alpha = [0.7];
pace = [0.01];
lambda = [100];
muta = 1000;
theta = [100];
query_size = 2100;
train_size = 10500;
% alpha = [0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
%   lambda = [0,0.01,0.1,1,10,100, 1000,10000, 100000];
%    muta = [0,0.1,1,10,100, 1000, 10000,100000];
% pace = [0.00005 0.0001 0.0005 0.001  0.005 0.01 0.05 0.1 0.5 1];
%  theta = [0,0.01,0.1,1,10,100, 1000,10000, 100000];

elseif strcmp(dataset_name, 'COCO')
alpha = [0.8];
pace = [0.1];
lambda = [100];
muta = 100;
theta = [0.001];
alpha = [0.9 1];
%   lambda = [0,0.01,0.1,1,10,100, 1000,10000, 100000];
%   pace = [ 0.1 0.5 1 5];
%  theta = [0.01,0.1,1,10,100, 1000,10000, 100000];
%    muta = [ 10000,100000];
X = double(I_db);
Y = T_db;
L = L_db;
query_size = 5000;
train_size = 10000;

elseif strcmp(dataset_name, 'IAPR12')
alpha = [0.8];
pace = [0.0001];
lambda = [1];
muta = 100;
theta = [1];
query_size = 2000;
train_size = 10000;
X = double(IAll');
Y = TAll';
L = LAll';
% alpha = [0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
%   lambda = [0.1,1,10,100, 1000,10000, 100000];
%     muta = [10,100, 1000, 10000,100000];
%   pace = [0.00005 0.0001 0.0005 0.001  0.005 0.01 0.05 0.1 0.5 1];
end  
%     clear X Y L PCA_Y R XAll
    %% Label Format
%     if isvector(LTrain)
%         LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
%         LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
%     end
 l =1;
    %% BATCH
for bi = 1:length(nbits)
    for j=1:length(lambda)
        for b = 1:length(pace)
           for k = 1:length(muta)
               for r=1:length(theta)
                  for q=1:length(alpha)
                    for v = 1:turn
                    R = randperm(size(X,1));
                    queryInds = R(1:query_size); 
                    sampleInds = R(query_size+1:query_size+train_size);
                    XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
                    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
                    maxItr = 15;
                    BMCHparam.dataset_name = dataset_name;
                    BMCHparam.alpha = alpha;
                    BMCHparam.pace = pace(b);
                    BMCHparam.lambda = lambda(j);
                    BMCHparam.muta = muta(k);
                    BMCHparam.theta= theta(r);
                    BMCHparam.nbits = nbits(bi);
                    BMCHparam.alpha = alpha(q);
                    BMCHparam.maxItr = maxItr;
                   
                    B = LERH(XTrain',YTrain',LTrain',BMCHparam);
                    eva_info_ = evaluate_BMCH(XTrain,YTrain,LTrain,XTest,YTest,LTest,BMCHparam,B);
                    eva_info_.Image_VS_Text_MAP
                    eva_info_.Text_VS_Image_MAP
                    result.bits = nbits(bi);
                    result.muta = muta(k);
                    result.theta =theta(r);
                    arry(l,1) = nbits(bi);
                    arry(l,3) = lambda(j);
                    arry(l,4) = muta(k);
                    arry(l,5) = theta(r);
                    arry(l,6) = eva_info_.Image_VS_Text_MAP;
                    arry(l,7) = eva_info_.Text_VS_Image_MAP;
                    arry(l,8) = eva_info_.Image_VS_Text_MAP + eva_info_.Text_VS_Image_MAP; 
                    l=l+1;
                     map(v,1)=eva_info_.Image_VS_Text_MAP;
                     map(v,2)=eva_info_.Text_VS_Image_MAP;
                     top(v,1) = eva_info_.I2Ttop;
                     top(v,2) = eva_info_.T2Itop;
                     
                    mapi2t(b,j)=mean(map( : , 1));
                    mapt2i(b,j)=mean(map( : , 2));
                % roWname={'bits','alpha','beta','lamda','Iter','i2t','t2i'};
                    end
             fprintf('%s : %d bits average map over %d runs for ImageQueryForText: %.4f, top@100 is %.4f\n',dataset_name,nbits, turn, mean(map( : , 1)),mean(top( : , 1)) );
             fprintf('%s : %d bits average map over %d runs for TextQueryForImage:  %.4f, top@100 is %.4f\n',dataset_name,nbits, turn, mean(map( : , 2)),mean(top( : , 2)));
                  end 
               end
              end
        end    
    end
%          xlswrite('mirflickr.xlsx',arry,'sheel1','A02');
%       save('I2T','mapi2t');
%       save('T2I','mapt2i');
end

end