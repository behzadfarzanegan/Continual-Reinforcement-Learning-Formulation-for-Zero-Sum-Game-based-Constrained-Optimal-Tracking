Neuron_Num_a =20;
m=1;
n=2;
Va = 1*randn(n*2+1,Neuron_Num_a);
Wa = 1*randn(Neuron_Num_a,m);

samples = 1000;
e2 = 4*rand(n,samples)-2;
ue = [1000];
r1=[0;0];
kk=1;
gamma=0.5;
while  norm(sum(abs(ue'))) > 3 && kk<5000
    for i = 1:samples
        if norm(sum(abs(ue'))) > 1
            alpha= 0.1;
        else
            alpha = 0.01;
        end
        augX = [e2(:,i);r1;1];
        dPol(:,i) = -2*e2(2,i);
        
        u_hat(:,i) = Wa'*logsig(Va'*augX);
        ue(:,i) = u_hat(:,i)-dPol(:,i);
        
        wG = logsig(Va'*augX);
        Wa = Wa-alpha*gamma*wG*ue(:,i)';
        Va= Va-alpha*(ue(:,i)'*Wa'.*(logsig(Va'*augX).*((1-logsig(Va'*augX)))*augX')');
    end
    norm(sum(abs(ue')))
    kk = kk + 1;
end
%
figure(10)
plot(u_hat')
hold on
plot(dPol','--')

v_actor(:,:,1)=Va;
W_actor(:,:,1)=Wa;


v_actor(:,:,2)=Va;
W_actor(:,:,2)=Wa;

