clear;close all;clc;
for j=1:1
    
 %initial NL   
% x(:,1) = [0.5 -0.5]'; 
% x(:,2) = [0.5 -0.5]';
% x(:,3) = [0.5 -0.5]';
%initial for sin
x(:,1) = 7*[0.1 0.1]'; 
x(:,2) = 7*[0.1 0.1]';
r(:,1) = [1 1]';
r(:,2) = [1 1]';
e(:,1)=x(:,1)-r(:,1);
e(:,2)=x(:,2)-r(:,2);
X(:,1) = [e(:,1);r(:,1)];
X(:,2) = [e(:,2);r(:,2)];

Q = 5*eye(2);
QN = [Q zeros(2,2);zeros(2,2) zeros(2,2)];
R = 1;


% Creat mesh domain
% aj =0.05;
aj =0.01;
gamma = 1.01;
%au = 0.001;

au = 0.01;
Neuron_Num_c = 11;
Neuron_Num_a = 11;
%second layer weight update
alpha1=0.01;
kv = 0.01;
kv1 = [0.01 0.01];
%B1 = 0.01*rand(Neuron_Num_c,1);
%B2 = 0.01*rand(Neuron_Num_a,1);
B1 = 0.01*rand(Neuron_Num_c,1);
B2 = 0.01*rand(Neuron_Num_a,1);
% %critic input layer v assumed contant
v_critic(:,:,1)= [0.0943,    0.3424,    0.6862,    0.0462,    0.8778,    0.8004,    0.7157,    0.5607,    0.6468,    0.3625,   0.1335 ;
    0.9300,    0.7360,    0.8936,    0.1955,    0.5824,    0.2859,    0.8390,    0.2691,    0.3077,    0.7881,  0.0216;
    0.3990,    0.7947,    0.0548,    0.7202,    0.0707,    0.5437,    0.4333,    0.7490,    0.1387,    0.7803,  0.5598;
    0.0474,    0.5449,    0.3037,    0.7218,    0.9227,    0.9848,    0.4706,    0.5039,    0.4756,    0.6685,  0.3008];
v_critic(:,:,2)= [0.9394,    0.8961,    0.5492,    0.4465,    0.9371,    0.5932,    0.2068,    0.6669,    0.7567,    0.8641,  0.7844;
    0.9809,    0.5975,    0.7284,    0.6463,    0.8295,    0.8726,    0.6539,    0.9337,    0.4170,    0.3889,    0.8828;
    0.2866,    0.8840,    0.5768,    0.5212,    0.8491,    0.9335,    0.0721,    0.8110,    0.9718,    0.4547,    0.9137;
    0.8008,    0.9437,    0.0259,    0.3723,   0.3725,    0.6685,    0.4067,    0.4845,    0.9880,    0.2467,     0.5583];

% %critic input layer v assumed random
% v_critic(:,:,1)=rand(4,Neuron_Num_c);
% v_critic(:,:,2)=rand(4,Neuron_Num_c);
% v_critic(:,:,3)=1*rand(4,Neuron_Num_c);

% optimal v critic with 21 neurons
%second layer of critic weights w constant
% W_critic(:,1) =[0.657530542095911;0.950915198264959;0.722348514327498;0.400079745362338;0.831871339329810;0.134338341728708;0.0604667719398284;0.0842470523146596;0.163898318329708;0.324219920294049;0.301726777204647];
% W_critic(:,2) = [0.0116809911303396;0.539905093849625;0.0953726926278216;0.146514856442232;0.631141207014955;0.859320411428785;0.974221631238713;0.570838427472184;0.996850214574949;0.553541573495533;0.515458454840326];

%second layer of critic weights w random
% W_critic(:,1) = 2*rand(Neuron_Num_c,1);
% W_critic(:,2) = 2*rand(Neuron_Num_c,1);
% W_critic(:,3) = 1*rand(Neuron_Num_c,1);

W_critic(:,1) = 1+3*rand(Neuron_Num_c,1);
W_critic(:,2) = W_critic(:,1);

% W_critic with 5 hidden neuros
%optimal v actor with 21 neurons
v_actor(:,:,1)=[0.8862,    0.8979,    0.8194,    0.4279,    0.7202,    0.1565,    0.8363,    0.3864,    0.6938,    0.1093,  0.0503;
    0.9311,    0.5934,    0.5319,    0.9661,    0.3469,    0.5621,    0.7314,    0.7756,   0.9452 ,   0.3899,   0.2287;
    0.1908,    0.5038,    0.2021,    0.6201,    0.5170,    0.6948,    0.3600,    0.7343,    0.7842,    0.5909,   0.8342;
    0.2586,    0.6128,    0.4539,    0.6954,    0.5567,    0.4265,    0.4542,    0.4303,    0.7056,    0.4594,  0.0156];
v_actor(:,:,2)=[0.8637,    0.2180,    0.5996,    0.0196,    0.5201,    0.1080,    0.0046,    0.9870,    0.5078,    0.6616, 0.5905; 
    0.0781,    0.5716,    0.0560,    0.4352,    0.8639,    0.5170,    0.7667,    0.5051,    0.5856,    0.5170,  0.4406;
    0.6690,    0.1222,    0.0563,    0.8322,    0.0977,    0.1432,    0.8487,    0.2714,    0.7629,    0.1710,  0.9419;
    0.5002,    0.6712,    0.1525,    0.6174,    0.9081,    0.5594,    0.9168,    0.1008,    0.0830,    0.9386,  0.6559];

 
% optimal v actor with 21 neurons random
% v_actor(:,:,1)= 1*rand(4,Neuron_Num_a);
% v_actor(:,:,2)= 1*rand(4,Neuron_Num_a);
% v_actor(:,:,3)= 1*rand(4,Neuron_Num_a);

% optimal W actor with 21 neurons

% W_actor(:,1)=[-0.0460029344953983;-0.465747005835051;0.0759474321256630;-0.918769644159625;-1.91946652982324;-0.0364431430103087;-1.22512571743544;-1.90243431779427;2.37424855083501;-0.233345713757840;0.403880739292361];
% W_actor(:,2) =[1.19246859933815;-1.68475884854101;0.413148529949933;0.501745219289013;0.0830643451642333;0.157798051973368;-0.527942514299507;0.723060699828803;-0.849941652073993;-0.796384050833210;0.725337571127988];

% optimal W actor with 21 neurons constant
% W_actor(:,1) = 1+1*randn(Neuron_Num_a,1);
% W_actor(:,2) = 1+1*randn(Neuron_Num_a,1);

W_actor(:,1) = 1+1*randn(Neuron_Num_a,1);
W_actor(:,2) = W_actor(:,1);

% W_actor(:,3) = 1*randn(Neuron_Num_a,1);
% u(1) = [0.1 0]*e(:,1);
% u(2) = [0.1 0]*e(:,2);

ad_u = 0.7;
u(1) = ad_u;
u(2) = ad_u;

J_sum(2)=0;
%x(:,2) = Linear_sys_std(x(:,1),u(1));
% Main Loop
for k = 2:1000
    % Sampling interval and basic variable initialization
    e(:,k)=x(:,k)-r(:,k);
    X(:,k) = [e(:,k);r(:,k)];
%     r(:,k+1)= [0.95*r(1,k) 1.2*r(2,k)];
    x(:,k)=[x(1,k);x(2,k)];
%     B = [0;0.2*x(2,k);0;0];  
    B = [0;0.2;0;0];  
    [x(:,k+1),r(:,k+1)] = NLinear_sys_NL_gamma_bah(x(:,k),u(k),r(:,k),k);
    e(:,k+1)=x(:,k+1)-r(:,k+1);
    X(:,k+1) = [e(:,k+1);r(:,k+1)];
    %J(k) = W_critic(:,k)'*Critic_NL_gamma_bah(X(:,k),v_critic(:,:,k),Neuron_Num_c);
    J(k) = (gamma^(-k))*(e(:,k)'*Q*e(:,k)+u(k)'*R*u(k));
    J_sum(k+1)=J_sum(k)+J(k);
%    
    for j = 1:1
        Yk_1 = [x(:,k-j)'*Q*x(:,k-j)+u(k-j)'*R*u(k-j)];
        delta_x = gamma*Critic_NL_gamma_bah(X(:,j+1),v_critic(:,:,k),Neuron_Num_c)-Critic_NL_gamma_bah(X(:,j),v_critic(:,:,k),Neuron_Num_c);
        Xk_1 = [delta_x];
    end
    EJK = Yk_1+W_critic(:,k)'*Xk_1;
%     YK = [Yk_1, x(:,k)'*Q*x(:,k)+u(k)'*R*u(k)];
    YK = [X(:,k)'*QN*X(:,k)+u(k)'*R*u(k)];
    temp_1 = v_critic(:,:,k);
%     temp_1 = v_critic(:,:,1);
    temp_2 = W_critic(:,k);
    for LL=1:10
        delta_xK = gamma*Critic_NL_gamma_bah(X(:,k+1),temp_1,Neuron_Num_c)-Critic_NL_gamma_bah(X(:,k),temp_1,Neuron_Num_c);
        XK = delta_xK;
    %     EJK = YK+W_critic(:,k)'*XK;
        %W_critic(:,k+1) = XK*inv(XK'*XK)*(aj*EJK'-YK');
       temp_1= temp_1-alpha1*X(:,k)*(temp_1'*X(:,k)+B1*kv*EJK)';
       temp_2 =temp_2-aj*(delta_xK*EJK)/(delta_xK'*delta_xK+1);
    end
    v_critic(:,:,k+1)=temp_1;
    W_critic(:,k+1)=temp_2;
    %B=[0;0.2;0;0];
    u_tilda = W_actor(:,k)'*Actor_NL_gamma_bah(X(:,k),v_actor(:,:,k),Neuron_Num_a)+gamma*0.5*inv(R)*B'*Dsigdx_NL_gamma_bah(X(:,k+1),v_critic(:,:,k),Neuron_Num_c)'*W_critic(:,k);
   
    temp = Actor_NL_gamma_bah(X(:,k),v_actor(:,:,k),Neuron_Num_a)/(Actor_NL_gamma_bah(X(:,k),v_actor(:,:,k),Neuron_Num_a)'*Actor_NL_gamma_bah(X(:,k),v_actor(:,:,k),Neuron_Num_a)+1);
%      W_actor(:,k+1) = W_actor(:,k)-au*temp*u_tilda';
%     v_actor1(:,:,k+1)=v_actor1(:,:,k)+(alpha1*X(:,k)*u_tilda')/(X(:,k)'*X(:,k)+1);
    
    W_actor(:,k+1) = W_actor(:,k)-au*temp*u_tilda';
    v_actor(:,:,k+1)=v_actor(:,:,k)+alpha1*X(:,k)*(v_actor(:,:,k)'*X(:,k)+B2*kv*u_tilda)';
%     u_tilda_error(:,k)=gamma*0.5*inv(R)*B'*Dsigdx_NL_gamma_bah(X(:,k+1),v_critic(:,:,k),Neuron_Num_c)'*W_critic(:,k);
    if k<200
        u(k+1) = W_actor(:,k+1)'*Actor_NL_gamma_bah(X(:,k+1),v_actor(:,:,k+1),Neuron_Num_a)+1*rand;
    else
        u(k+1) =W_actor(:,k+1)'*Actor_NL_gamma_bah(X(:,k+1),v_actor(:,:,k+1),Neuron_Num_a);
    end
    U_tilda(k)=u_tilda;
    Err(k)=EJK;
    cost(k)=YK;
end

figure(1);hold on;
% nn = 1:iter+1;
subplot 331; plot(x(1,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(r(1,:),'--r','LineWidth',2);
ylabel('x1,r1','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
title('x1 and r1','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 332; plot(x(2,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(r(2,:),'--r','LineWidth',2);
ylabel('x2,r2','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
title('x2 and r2 ','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 333; plot(e(1,:),'--r','LineWidth',2);hold on;plot(e(2,:),'--b','LineWidth',2);hold on;
%plot(J_sum,'k','LineWidth',2);
% hold on;
xlabel('Sampling instants','FontWeight','b','FontSize',12);
 ylabel('e_1 and e_2','FontWeight','b','FontSize',12);
 title('e1 and e2 variation','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 334;
for i = 1:11
     hold on;
     plot(W_critic(i,:),'LineWidth',2);
 end
 xlabel('Sampling instants','FontWeight','b','FontSize',12);
 ylabel('W_{critic}','FontWeight','b','FontSize',12);
 title('Weights variation','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 335; plot(Err,'b');hold on;
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('Err ','FontWeight','b','FontSize',12);
title('Error','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 336;
plot(U_tilda,'r');
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('utilda','FontWeight','b','FontSize',12);
title('Utilda','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 337;
plot(u,'r');
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('\hat{u}','FontWeight','b','FontSize',12);
title('\hat{u}','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 338;
plot(J_sum,'r');
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('Estimated Cost','FontWeight','b','FontSize',12);
title('Estimated Cost','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

set(gcf,'units','normalized','outerposition',[0 0 1 1])

set(gcf,'units','normalized','outerposition',[0 0 1 1])


% figure;
% plot(u,'r');
% xlabel('Sampling instants','FontWeight','b','FontSize',12);
% ylabel('$$\hat{u}$$','FontWeight','b','FontSize',12,'Interpreter','latex');
% title('$$\hat{u}$$','FontWeight','b','FontSize',12,'Interpreter','latex');
% grid on;box on;
% set( gca, 'FontWeight', 'b','FontSize', 12 );
% 
% % creating the zoom-in inset
% ax=axes;
% set(ax,'units','normalized','position',[0.4,0.2,0.2,0.2]);
% box(ax,'on');
% plot(u,'r','LineWidth',2,'parent',ax); grid on;
% set(ax,'xlim',[800,850],'ylim',[-0.1,0.1]);
end










