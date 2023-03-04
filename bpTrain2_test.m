Vc = 1+5*rand(n*2,Neuron_Num_c);
Wc = 2*rand(Neuron_Num_c,1);
Vc2 = Vc;
Wc2 = Wc;
Va = 10000*randn(n*2,Neuron_Num_a);
Wa = 10+10*randn(Neuron_Num_a,m);

samples = 1000;
e2 = 1*randn(n,samples);
% ue = [1000;1000];
ue=[1001];
% r = [0;0;0;0;1];
r1=[0;0];
% load('WTS')
while  norm(sum(abs(ue'))) > 135
    for i = 1:samples
        if norm(sum(abs(ue'))) > 150
            
            alpha= 0.1;
        else
            alpha = 0.00001;
        end
        augX = [e2(:,i);r1];
%         dPol(:,i) = -[100,0,20,0;0,100,0,20]*e2(:,i);
%             dPol(:,i) = 5*sin(e2(2,i));%initial controller
            dPol(:,i) = (e2(2,i)^2);%initial controller
        u_hat(:,i) = Wa'*tanh(Va'*tanh(augX));
        ue(:,i) = u_hat(:,i)-dPol(:,i);
        
        % GD alg
%         Wa = Wa-alpha*gamma*tanh(Va'*augX)*ue(:,i)';
%         Va= Va-alpha*gamma*ue(:,i)'*Wa'.*((1-tanh(Va'*augX).^2)*augX')';
%         % Newton Alg
         wG = tanh(Va'*tanh(augX));
        Wa = Wa-alpha*gamma*wG*ue(:,i)';

        Va= Va-alpha*gamma*(ue(:,i)'*Wa'.*((1-tanh(Va'*tanh(augX)).^2)*tanh(augX)')');

% Wa = Wa-1*alpha*gamma*wG*ue(:,i)';
%         Va= Va-alpha*tanh(augX)*(Va'*tanh(augX)+B2*kv*ue(:,i))';
    end
    norm(sum(abs(ue')))
    
end
% 
figure(10)
plot(u_hat')
hold on
plot(dPol','--')
t=0:.01:10;
for i=1:1001
    fun(i)=Wa'*tanh(Va'*tanh([0;t(i);0;0]));
end
plot(t,fun)
% %% Train critic
 tde = 10001;
 x2 = 4*randn(n,2*samples);
 e2 = x2-r1(1:n);
% % Q = 0.1*eye(4);
% % R = 0.01*eye(2);
% % load('cWTS')
while  norm(tde') > 1450 %Q=50 7600
x2 = 4*rand(n,2*samples)-2;
 e2 = x2-r1(1:n);
    for i = 1:samples*2
        
%         if norm(sum(abs(ue'))) > 1000
%             alpha= 0.00001;
%         else
%             alpha = 0.000001;
%         end
        augX = [e2(:,i);r1];
        zeroIn = [zeros(n*2,1)];
        u_hat(:,i) = Wa'*tanh(Va'*tanh(augX));
%         trns = robDynamics(x2(:,i),u_hat(:,i));
        [trns,r2,Gx] = NLinear_sys_NL_gamma_bah(x2(:,i),u_hat(:,i),r1,i);
        ek1 = trns-r1(1:n);
        augXk1 = [ek1;r1];
        
              
            
            J = Wc'*tanh(Vc'*tanh(augXk1));
            ij = tanh(Vc'*tanh(augXk1));
            Jp= Wc'*tanh(Vc'*tanh(augX));
            ijp= tanh(Vc'*tanh(augX));
            

            
            %train net1
%             Wc = Wc-0.01*gamma*er1*tanh(Vc'*tanh(augX));
%             Vc= Vc-0.01*gamma*er1*Wc'.*(sech(Vc'*tanh(augX)).^2'.*tanh(augX));
%         
%         
        
        L = @(x_L,u_L) x_L'*Q*x_L + u_L'*R*u_L;
      

        
        dj = L(e2(:,i),u_hat(:,i));
        deltaSig = gamma*ij-ijp;
        
        tde(i) = dj+Wc'*deltaSig;
        
        aj = 0.001;
        % GD alg
%         Wa = Wa-alpha*gamma*tanh(Va'*augX)*ue(:,i)';
%         Va= Va-alpha*gamma*ue(:,i)'*Wa'.*((1-tanh(Va'*augX).^2)*augX')';
        % Newton Alg
        delW = aj*deltaSig*tde(i);
        delV = Wc'.*(gamma*(sech(Vc'*tanh(augXk1)).^2'.*tanh(augXk1))-(sech(Vc'*tanh(augX)).^2'.*tanh(augX)));
        Wc = Wc-delW;
        Vc= Vc-aj*delV*tde(i);
    end
     norm(tde)
 end


% %% algorithm from paper: https://ieeexplore-ieee-org.libproxy.mst.edu/stamp/stamp.jsp?tp=&arnumber=6609085
% u = zeros(m,1);
%  r11 = [0;0];
% [output1,output2,G] = NLinear_sys_NL_gamma_bah(x(:,2),u,r11,2);
% L = @(x_L,u_L) x_L'*QN*x_L + u_L'*R*u_L;
% x1=e2(:,i);
% augX = [e2(:,i);r1];
% for i = 1:35
%     if i ==1
%         %         u = Wa'*tanh(Va'*augX);
%         er1 =1;
%         ue = 0;
%         while abs(er1+ue)>0.01
%             phi = L(augX,u);
%             net1= Wc'*tanh(Vc'*tanh(augX));
%             u = Wa'*tanh(Va'*tanh(augX));
%             er1 = net1-phi;
% %             derivTerm =(eye(Neuron_Num_c)-diag(tanh(Vc'*tanh(X(:,2))).^2))*Vc'*(eye(length(augX))-diag(tanh(augX).^2));
%             %             derivTerm = Vc'.*repmat(1-tanh(Vc'*tanh(X(:,2))).^2,1,n*2);
% %             ue = u+gamma/2*inv(R)*G'*derivTerm'*Wc;
%             %train net1
%             Wc = Wc-0.01*gamma*er1*tanh(Vc'*tanh(augX));
%             Vc = Vc-0.01*gamma*er1*Wc'.*(sech(Vc'*tanh(augX)).^2'.*tanh(augX));
%             
% %             %train u
% %             Wa = Wa-0.01*gamma*tanh(Va'*tanh(augX))*ue';
% %             Va= Va-0.01*gamma*ue'*Wa'.*((1-tanh(Va'*tanh(augX)).^2)*tanh(augX)')';
%             
%         end
%         %copy to net 2
%         Vc2 = Vc;
%         Wc2 = Wc;
%     else
%         i;
%         er1;
%          er1 =1;
%         ue=0;
%          while abs(er1+ue)>0.001
%             [xk1,r2,Gx] = NLinear_sys_NL_gamma_bah(x1,u,r11,i);
%             
%             augX1 = [xk1-r2(1:n);r2];
% %             x1=e2;
%             augX = [x1-r11;r11];
%             net2 = Wc2'*tanh(Vc2'*tanh(augX1));
%             u = Wa'*tanh(Va'*tanh(augX));
%             phi = L(augX,u)+gamma*net2;
%             net1= Wc'*tanh(Vc'*tanh(augX));
%             derivTerm = Vc'.*repmat(1-tanh(Vc'*tanh(augX)).^2,1,n*2);
% %             ue = u+gamma/2*inv(R)*G'*derivTerm'*Wc;
%             
%             er1 = net1-phi;
%             
%             %train net1
%             Wc = Wc-0.1*gamma*er1*tanh(Vc'*tanh(augX));
%             Vc= Vc-0.1*gamma*er1*Wc'.*(sech(Vc'*tanh(augX)).^2'.*tanh(augX));
%             
% %             %train u
% %             Wa = Wa-0.01*gamma*tanh(Va'*tanh(augX))*ue';
% %             Va= Va-0.01*gamma*ue'*Wa'.*((1-tanh(Va'*tanh(augX)).^2)*tanh(augX)')';
%          end
%         %copy to net 2
%         %         net1-Wc2'*tanh(Vc2'*augX)
%         if i<50
%             Vc2 = Vc;
%             Wc2 = Wc;
%             x1=xk1;
% %             augX = augX1;
% %             x1 = augX(1:n);
% %             u = Wa'*tanh(Va'*tanh(augX));
% %             r11=r2;
%         end
%         
%     end
%     
% end

v_actor(:,:,1)=Va;
W_actor(:,1)=Wa;
v_critic(:,:,1)=.14*Vc;
W_critic(:,1)=-20*Wc;

v_actor(:,:,2)=Va;
W_actor(:,2)=Wa;
v_critic(:,:,2)=v_critic(:,:,1);
W_critic(:,2)=W_critic(:,1);
