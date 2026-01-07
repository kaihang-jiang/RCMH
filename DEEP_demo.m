close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

tic;
dataset_name = {'NUSWIDE_clip'};%MIRFLICKR NUSWIDE
param.top_K = 2000;
for dbi = 1:length(dataset_name)
    dataset_name = dataset_name{dbi}; param.dataset_name = dataset_name;
    turn = 5;           
    nbits = [32];
    cbits = [128];
    %% load dataset
% load(['./datasets/',dataset_name,'_vgg.mat']) % for vgg features
load(['./datasets/',dataset_name,'.mat'])

       if strcmp(dataset_name, 'MIRFLICKR_vgg')
            alpha = [0.6];
            pace = [0.05];
            muta = 1;
            theta = [1000];
            query_size = 2000;
            train_size = 10000;


        elseif strcmp(dataset_name, 'MIRFLICKR_clip')
            alpha = [0.6];
            pace = [0.005];
            muta = [1];
            theta = [1];
            I_tr = full(double(I_tr));
            I_te = full(double(I_te));
            T_tr = full(double(T_tr));
            T_te = full(double(T_te));
            L_tr = full(double(L_tr));
            L_te = full(double(L_te));
            XTrain = I_tr;
            YTrain = T_tr;
            LTrain = L_tr;
            XTest = I_te;
            YTest = T_te;
            LTest = L_te;

        elseif strcmp(dataset_name, 'NUSWIDE_vgg')
            alpha = [0.6];
            pace = [0.05];
            lambda = [100];
            muta = 100;
            theta = [100];
            query_size = 2100;
            train_size = 10500;

        elseif strcmp(dataset_name, 'NUSWIDE_clip')
            alpha = [1];
            pace = [0.5];
            % pace = [0.0005 0.001  0.005 0.01 0.05 0.1 0.5 1];
            lambda = [1000];
            mu = [1];
            theta = [1];
            I_tr = full(double(I_tr));
            I_te = full(double(I_te));
            T_tr = full(double(T_tr));
            T_te = full(double(T_te));
            L_tr = full(double(L_tr));
            L_te = full(double(L_te));
            XTrain = I_tr;
            YTrain = T_tr;
            LTrain = L_tr;
            XTest = I_te;
            YTest = T_te;
            LTest = L_te;

        elseif strcmp(dataset_name, 'coco_vgg')
            alpha = [0.6];
            lambda = [100];
            muta = 100;
            theta = [100];
            pace = [0.05];

            X = [I_db;I_te];
            Y = [T_db;T_te];
            L = [L_db;L_te];


        elseif strcmp(dataset_name, 'coco')
            alpha = [1];
            pace = [0.1];
            muta = [1];
            theta = [1];
            query_size = 2000;
            train_size = 10000;
            I_tr = full(double(I_tr));
            I_te = full(double(I_te));
            T_tr = full(double(T_tr));
            T_te = full(double(T_te));
            L_tr = full(double(L_tr));
            L_te = full(double(L_te));
            XTrain = I_tr;
            YTrain = T_tr;
            LTrain = L_tr;
            XTest = I_te;
            YTest = T_te;
            LTest = L_te;

        elseif strcmp(dataset_name, 'IAPRTC12_vgg')
            alpha = [0.12];
            pace = [0.01];
            muta = 1000;
            theta = [10];
            query_size = 2000;
            train_size = 10000;
            X = double(IAll');
            Y = TAll';
            L = LAll';
  
        elseif strcmp(dataset_name, 'IAPRTC12_clip')
            alpha = [0.9];
            pace = [0.005];
            lambda = [10];
            muta = [1];
            theta = [10];
            I_tr = full(double(I_tr));
            I_te = full(double(I_te));
            T_tr = full(double(T_tr));
            T_te = full(double(T_te));
            L_tr = full(double(L_tr));
            L_te = full(double(L_te));
            XTrain = I_tr;
            YTrain = T_tr;
            LTrain = L_tr;
            XTest = I_te;
            YTest = T_te;
            LTest = L_te;
         end  
    %% RCMH deep
    for bi = 1:length(nbits)
         for u = 1:length(cbits)
            for b = 1:length(pace)
                 for k = 1:length(mu)
                   for r=1:length(theta)
                      for q=1:length(alpha)
                        for v = 1:turn
                        maxItr = 1;
                        BMCHparam.dataset_name = dataset_name;
                        BMCHparam.alpha = alpha;
                        BMCHparam.pace = pace(b);
                        BMCHparam.mu = mu(k);
                        BMCHparam.theta= theta(r);
                        BMCHparam.nbits = nbits(bi);
                        BMCHparam.alpha = alpha(q);
                        BMCHparam.maxItr = maxItr;
                        BMCHparam.cbits = cbits(u);

                        B = RCMH(XTrain',YTrain',LTrain',v,BMCHparam);
                        eva_info_ = evaluate_RCMH(XTrain,YTrain,LTrain,XTest,YTest,LTest,BMCHparam,B);
                        eva_info_.Image_VS_Text_MAP;
                        eva_info_.Text_VS_Image_MAP;
                         map(v,1)=eva_info_.Image_VS_Text_MAP;
                         map(v,2)=eva_info_.Text_VS_Image_MAP;
                         top(v,1) = eva_info_.I2Ttop;
                         top(v,2) = eva_info_.T2Itop;

                        end
                 fprintf('%s : %d bits average map over %d runs for ImageQueryForText: %.4f, top@100 is %.4f\n',dataset_name,cbits, turn, mean(map( : , 1)),mean(top( : , 1)) );
                 fprintf('%s : %d bits average map over %d runs for TextQueryForImage:  %.4f, top@100 is %.4f\n',dataset_name,cbits, turn, mean(map( : , 2)),mean(top( : , 2)));
                      end
                    end
                end
            end    
         end
    end

end