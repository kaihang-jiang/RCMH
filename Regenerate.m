function [B,U] =Regenerate(XKTrain,YKTrain,B,U,G,param)
    lambda = param.lambda;
    pace = param.pace;
    muta = param.muta;
    [nbits, n] = size(B);
    d1 = size(XKTrain,1);
    d2 = size(YKTrain,1);
    H_1 = randn(nbits,d1);
    H_2 = randn(nbits,d2);
    E_1 = diag(rand(2*nbits,1));
     for iter = 1: 400
        deriva_1  = -((B-tanh(H_1*XKTrain))).*(ones(nbits, n)-tanh(H_1*XKTrain).*tanh(H_1*XKTrain))*XKTrain';
        H_1 = H_1- pace*deriva_1;
        deriva_2  = -((B-tanh(H_2*YKTrain))).*(ones(nbits, n)-tanh(H_2*YKTrain).*tanh(H_2*YKTrain))*YKTrain';
        H_2 = H_2- pace*deriva_2;
     end
     
     Bx = sign(H_1*XKTrain);
     By = sign(H_2*YKTrain);
     [~,~,Y] = svd([Bx;By],'econ');

     % update W_t
     [C_1,~,F_1] = svd(E_1*Y'*B','econ'); 
     W_1 = C_1*F_1';
        
     %update E_t
        J_1 = Y'; 
        K_1 =W_1 *B;
        for j = 1:2*nbits
            if norm(J_1(j,:))~=0 && norm(K_1(j,:))~=0
            e_1(j) = J_1(j,:)*K_1(j,:)'/(norm(J_1(j,:))*norm(K_1(j,:)));
            else
            e_1(j) = J_1(j,:)*K_1(j,:)';
            end
        end
        
%           a = median(e_1,'all');
        a = mean(e_1);
        e_1(e_1>=a) = 1;
        e_1(e_1<a) = -1;
        E_1 =diag(e_1);
        

        % B-step
        B = sign((Bx*Bx'+By*By'+(lambda+muta)*eye(nbits))\(nbits*((Bx+By)*G')*G + lambda*(W_1'*E_1*Y') + muta* U));
        B(B==0) = -1;
end