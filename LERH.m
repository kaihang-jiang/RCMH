function final_B = LERH(XKTrain,YKTrain,LTrain,param)
  %%  parameters
      lambda = param.lambda;
      nbits = param.nbits;
      name = param.dataset_name;
      muta = param.muta;
      pace = param.pace;
      XKTrain = double(XKTrain);
      G = label_enhance(XKTrain, YKTrain, LTrain, param);

      [c,n] = size(LTrain);
      [d1,~] = size(XKTrain);
      [d2,~] = size(YKTrain);
  %%  initial
    B = sign(randn(nbits,n));  B(B==0) = -1;
    U = B;
    E_1 = diag(rand(2*nbits,1));
    H_1 = randn(nbits,d1);
    H_2 = randn(nbits,d2);
    for i = 1:param.maxItr
        % update V
        V = (nbits*U*G'*G )';
        Temp = V'*V-1/n*(V'*ones(n,1)*(ones(1,n)*V));
        [~,Lmd,RR] = svd(Temp);
        idx = (diag(Lmd)>1e-8);
        R = RR(:,idx); R_ = orth(RR(:,~idx));
        P = (V-1/n*ones(n,1)*(ones(1,n)*V)) *  (R / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n,nbits-length(find(idx==1))));
        V = (sqrt(n)*[P P_]*[R R_]')'; 
        clear idx RR Lmd Temp
        
        
        % update U
        U = sign((nbits*V*G'*G + muta* B)/(n+muta)); 
        U(U==0) = -1;  

        B = U; 
        % update H
        for iter = 1: 60
        deriva_1  = -((V-tanh(H_1*XKTrain))).*(ones(nbits, n)-tanh(H_1*XKTrain).*tanh(H_1*XKTrain))*XKTrain';
        H_1 = H_1- pace*deriva_1;
        deriva_2  = -((V-tanh(H_2*YKTrain))).*(ones(nbits, n)-tanh(H_2*YKTrain).*tanh(H_2*YKTrain))*YKTrain';
        H_2 = H_2- pace*deriva_2;
        end
        clear deriva_1 deriva_2

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

        a = mean(e_1);
        e_1(e_1>=a) = 1;
        e_1(e_1<a) = -1;
        E_1 =diag(e_1);
        

        % B-step
        B = sign((Bx*Bx'+By*By'+(lambda+muta)*eye(nbits))\(nbits*((Bx+By)*G')*G + lambda*(W_1'*E_1*Y') + muta* U));
        B(B==0) = -1;
     

        U = B;
        
    end
       final_B  = sign(B);final_B(final_B==0)=-1;
       clear H_1 deriva_1 H_2 deriva_2 B U
    end

     