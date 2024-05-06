clc;
clear;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
dataset_name = {'coco'};
%   rng('default');
%% load dataset
 for db = 1:length(dataset_name)
  dataset = dataset_name{db};
  load(['./datasets/',dataset,'.mat']);
%% parameter setting 
 nbits = [128];
 alpha = [0.8];
  pace = [0.1];
lambda = [10000];
  muta = 10000;
 theta = [10];
maxItr = 10;
  turn = 10; %turns
  func = 'linear';
     l = 1; %excel writing parameter
 %% start
for bi = 1:length(nbits)
    for j=1:length(lambda)
       for ai = 1:length(alpha)
        for b = 1:length(pace)
           for k = 1:length(muta)
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
                    BMCHparam.lambda = lambda(j);
                    BMCHparam.muta = muta(k);
                    BMCHparam.theta= theta(r);
                    BMCHparam.nbits = nbits(bi);
                    BMCHparam.maxItr = maxItr;
                    BMCHparam.func = func;
                   
                    B = LERH(XTrain',YTrain',LTrain',BMCHparam);
                    eva_info_ = evaluate_LERH(XTrain,YTrain,LTrain,XTest,YTest,LTest,BMCHparam,B);
                    eva_info_.Image_VS_Text_MAP
                    eva_info_.Text_VS_Image_MAP
                    result.bits = nbits(bi);
                    result.muta = muta(k);
                    result.theta =theta(r);
                % roWname={'bits','alpha','beta','lamda','Iter','i2t','t2i'};
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
                     
                    mapi2t(ai,1)=mean(map( : , 1));
                    mapt2i(ai,1)=mean(map( : , 2));
                % roWname={'bits','alpha','beta','lamda','Iter','i2t','t2i'};
                  end
             fprintf('%d bits average map over %d runs for ImageQueryForText: %.4f\n, top@100 is %.4f\n',nbits, turn, mean(map( : , 1)),mean(top( : , 1)) );
             fprintf('%d bits average map over %d runs for TextQueryForImage:  %.4f\n, top@100 is %.4f\n',nbits, turn, mean(map( : , 2)),mean(top( : , 2)));
               end 
           end
        end
      end    
    end
%          xlswrite('mirflickr.xlsx',arry,'sheel1','A02');
      save('I2T','mapi2t');
      save('T2I','mapt2i');
end
end