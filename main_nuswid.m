clc;
clear;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));
dataset_name = 'NUSWIDE10';
%   rng('default');
%% load dataset

load(['./datasets/',dataset_name,'.mat']); 
%% parameter setting 
nbits = 32; % bit length of existing hash codes
cbits = [32]; % bit length of learned hash codes
 
alpha = 0.6;
pace = 0.005;
mu = 10;
theta = 10;
maxItr = 1;
turn = 5; %turns
l = 1; %excel writing parameter
 %% start
for bi = 1:length(nbits)
    for j=1:length(cbits)
        for b = 1:length(pace)
           for k = 1:length(mu)
             for ai = 1:length(alpha)
               for r=1:length(theta)
                  for v = 1:turn
                    BMCHparam.dataset_name = dataset_name;
                    XTrain = I_tr;
                    YTrain = T_tr;
                    LTrain = L_tr;
                    XTest = I_te;
                    YTest = T_te;
                    LTest = L_te;
                    BMCHparam.alpha = alpha(ai);
                    BMCHparam.pace = pace(b);
                    BMCHparam.cbits = cbits(j);
                    BMCHparam.mu = mu(k);
                    BMCHparam.theta= theta(r);
                    BMCHparam.nbits = nbits(bi);
                    BMCHparam.maxItr = maxItr;
                    B = RCMH(XTrain',YTrain',LTrain',v,BMCHparam);
                    eva_info_ = evaluate_RCMH(XTrain,YTrain,LTrain,XTest,YTest,LTest,BMCHparam,B);
                    map(v,1)=eva_info_.Image_VS_Text_MAP;
                    map(v,2)=eva_info_.Text_VS_Image_MAP;
                    top(v,1) = eva_info_.I2Ttop;
                    top(v,2) = eva_info_.T2Itop;
                  end
                   fprintf('%d bits to %d bits average map over %d runs for ImageQueryForText: %.4f\n, top@100 is %.4f\n',nbits, cbits(j), turn, mean(map( : , 1)),mean(top( : , 1)) );
                   fprintf('%d bits to %d bits average map over %d runs for TextQueryForImage:  %.4f\n, top@100 is %.4f\n',nbits, cbits(j), turn, mean(map( : , 2)),mean(top( : , 2)));
               end 
           end
        end 
      end
    end

end