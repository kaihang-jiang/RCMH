function final_B = RCMH(XKTrain,YKTrain,LTrain,v,param)
  %%  parameters
      nbits = param.nbits;
      XKTrain = double(XKTrain);
     [~,n] = size(LTrain);
  %% random initial existing hash codes U
    U = sign(normrnd(0, 1, nbits,n));
    U(U==0)=-1;
    
    tic;
    [B] = Regenerate(XKTrain,YKTrain,LTrain,U,param);
    fprintf('Training time of %d turn is %.2f s\n',v,toc);
    
    final_B  = sign(B);final_B(final_B==0)=-1;
end