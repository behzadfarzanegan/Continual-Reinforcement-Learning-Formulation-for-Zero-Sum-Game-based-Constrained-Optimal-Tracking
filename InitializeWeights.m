Vc = 2*rand(n*2,Neuron_Num_c)-1;
Wc = 2*rand(Neuron_Num_c,1)-1;
Vc2 = Vc;
Wc2 = Wc;
Va = rand(n*2,Neuron_Num_a)-.5;
Wa = rand(Neuron_Num_a,m)-.5;

xLen = length(x);
%% algorithm from paper: https://ieeexplore-ieee-org.libproxy.mst.edu/stamp/stamp.jsp?tp=&arnumber=6609085
u = zeros(m,1);
 r11 = r(:,2);
[output1,output2,G] = NLinear_sys_NL_gamma_bah(x(:,2),u,r11,2);
L = @(x_L,u_L) x_L'*QN*x_L + u_L'*R*u_L;
x1=x(:,2);
augX=X(:,2);
for i = 1:35
    if i ==1
        %         u = Wa'*tanh(Va'*augX);
        er1 =1;
        ue = 1;
        while abs(er1+ue)>0.01
            phi = L(augX,u);
            net1= Wc'*tanh(Vc'*tanh(augX));
            u = Wa'*tanh(Va'*tanh(augX));
            er1 = net1-phi;
            derivTerm =(eye(Neuron_Num_c)-diag(tanh(Vc'*tanh(X(:,2))).^2))*Vc'*(eye(length(augX))-diag(tanh(augX).^2));
            %             derivTerm = Vc'.*repmat(1-tanh(Vc'*tanh(X(:,2))).^2,1,n*2);
            ue = u+gamma/2*inv(R)*G'*derivTerm'*Wc;
            %train net1
            Wc = Wc-0.01*gamma*er1*tanh(Vc'*tanh(augX));
            Vc = Vc-0.01*gamma*er1*Wc'.*(sech(Vc'*tanh(augX)).^2'.*tanh(augX));
            
            %train u
            Wa = Wa-0.01*gamma*tanh(Va'*tanh(augX))*ue';
            Va= Va-0.01*gamma*ue'*Wa'.*((1-tanh(Va'*tanh(augX)).^2)*tanh(augX)')';
            
        end
        %copy to net 2
        Vc2 = Vc;
        Wc2 = Wc;
    else
        er1 =1;
        ue=1;
        while abs(er1+ue)>0.01
            [xk1,r2,Gx] = NLinear_sys_NL_gamma_bah(x1,u,r11,i);
            
            augX1 = [xk1-r2(1:n);r2];
            
            net2 = Wc2'*tanh(Vc2'*tanh(augX1));
            u = Wa'*tanh(Va'*tanh(augX));
            phi = L(augX,u)+gamma*net2;
            net1= Wc'*tanh(Vc'*tanh(augX));
            derivTerm = Vc'.*repmat(1-tanh(Vc'*tanh(augX)).^2,1,n*2);
            ue = u+gamma/2*inv(R)*G'*derivTerm'*Wc;
            
            er1 = net1-phi;
            
            %train net1
            Wc = Wc-0.01*gamma*er1*tanh(Vc'*tanh(augX));
            Vc= Vc-0.01*gamma*er1*Wc'.*(sech(Vc'*tanh(augX)).^2'.*tanh(augX));
            
            %train u
            Wa = Wa-0.01*gamma*tanh(Va'*tanh(augX))*ue';
            Va= Va-0.01*gamma*ue'*Wa'.*((1-tanh(Va'*tanh(augX)).^2)*tanh(augX)')';
        end
        %copy to net 2
        %         net1-Wc2'*tanh(Vc2'*augX)
        if i<50
            Vc2 = Vc;
            Wc2 = Wc;
%             augX = augX1;
%             x1 = augX(1:n);
%             u = Wa'*tanh(Va'*tanh(augX));
%             r11=r2;
        end
        
    end
    
end
v_actor(:,:,1)=Va;
W_actor(:,1)=Wa;
v_critic(:,:,1)=Vc;
W_critic(:,1)=Wc;

v_actor(:,:,2)=Va;
W_actor(:,2)=Wa;
v_critic(:,:,2)=Vc;
W_critic(:,2)=Wc;