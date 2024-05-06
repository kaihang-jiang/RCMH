function [G] = label_enhance(X,Y,L,param)
[c,n]= size(L);
       X = bsxfun(@minus, X, mean(X, 1)); 
       X = NormalizeFea(X,0);

       Y = bsxfun(@minus, Y, mean(Y, 1)); 
       Y = NormalizeFea(Y,0);
       
       f = sum(L,2); 
       a = median(f,'all');
       f = f/a;
       f = 1 ./ (1 + exp(-f));
       T = NormalizeFea(L.*f,0);
alpha = param.alpha;
M = rand(c,n);
   for i=1 : 100
       %update Q_t 
       temp_1 = (X*(M.*L)');
       [U_1,~,V_1] = svd(temp_1,'econ');
       Q_1 = U_1*V_1';
       temp_2 = (Y*(M.*L)');
       [U_2,~,V_2] = svd(temp_2,'econ');
       Q_2 = U_2*V_2';
       temp_3 = (T*(M.*L)');
       [U_3,~,V_3] = svd(temp_3,'econ');
       Q_3 = U_3*V_3'; 
       M = ((1-alpha)/2*Q_1'*X+(1-alpha)/2*Q_2'*Y+alpha*Q_3'*T).*L;
       C =  M(M<0) ;
       M(M<0) = -C;
       M = NormalizeFea(M,0);
   end
   G = NormalizeFea(M.*L,0);
end