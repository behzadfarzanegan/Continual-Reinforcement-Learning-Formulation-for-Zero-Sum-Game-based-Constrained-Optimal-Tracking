clear all;
close all;clc;

% format short

x(:,1) = [0,1]';
x(:,2) = [0,1]';
r(:,1) = [0;1]';
r(:,2) = [0;1]';
e(:,1)=x(:,1)-r(:,1);
e(:,2)=x(:,2)-r(:,2);
X(:,1) = [e(:,1);r(:,1);1];
X(:,2) = [e(:,2);r(:,2);1];
AugX(:,1) = [e(:,1);r(:,1)];
AugX(:,2) = [e(:,2);r(:,2)];

n=length(x);%states
m=1;%inputs
Q = 5*eye(n);
QN = [Q zeros(n,n+1);zeros(n+1,n) zeros(n+1,n+1)];
R = .010*eye(m);


% Creat mesh domain
% aj =0.05;
gamma =.5;
gamma2 =.1;
H=[0 1 0 0]';
%au = 0.001;


%Neuron_Num_c = sum(1:length(AugX));
Neuron_Num_c = 27;
Neuron_Num_a =20;
Neuron_Num_a2 =20;
%second layer weight update.
S = 300;

Q11=[];
Q22=[];
Q33=[];
Q44=[];

%%
%   bpTrain
% load('WEIHGT.mat')
%  InitializeWeights;

%    load('WcVc.mat')



load('WaVa.mat')
v_actor(:,:,1)=Va;
W_actor(:,:,1)=Wa;


Wc = 1*rand(Neuron_Num_c,1);
load('Wc.mat')
W_critic(:,:,1)=Wc;


v_actor(:,:,2)=v_actor(:,:,1);
W_actor(:,:,2)=W_actor(:,:,1);




W_critic(:,:,2)=W_critic(:,:,1);



Vw =1*randn(2*n+1,Neuron_Num_a2);
Ww = 1*randn(Neuron_Num_a2,m);
% load('WEIHGTVwWw')
v_actor2(:,:,1)=Vw;
W_actor2(:,:,1)=Ww;
v_actor2(:,:,2)=v_actor2(:,:,1);
W_actor2(:,:,2)=W_actor2(:,:,1);


aj =0.01;

au = 0.4*1;


aw = 0.0003;


L=10;PE_C=1;M1=.5;
%%



ad_u = W_actor(:,:,2)'*logsig(v_actor(:,:,2)'*X(:,2));
% ad_u=[0.7;0.7];
u(:,1) = ad_u;
u(:,2) = ad_u;
%
% J_sum(1:2)=0;
Jhatsum(1:2)=0;
%x(:,2) = Linear_sys_std(x(:,1),u(1));
% Main Loop


svMaxing = true;
expReplay = true;
Z= zeros(2*n,2);
p=0; %samples in replay buffer
buffSize = 36; %size of the buffer
dphi=[];
disturbance(1:2000)=0;
ad_w=[0.1;0.1];
w(:,1)=ad_w;
w(:,2)=ad_w;
dist_on=.5;
Barrier = true;
T=0.01;
BF=0;
safety=false;
EWC=false;
for k = 2:10000
      if k==3002
        e(:,k)=0;
        x(2,k)=r(2,k);
        AugX(:,k) = [e(:,k);r(:,k)];
      end
        if k==6002
        e(:,k)=0;
        x(2,k)=r(2,k);
        AugX(:,k) = [e(:,k);r(:,k)];
    end


    %u(:,k)=-2*e(2,k);

    %%


    %%
    %      if k>200
    BF1 = log((-0.3*0.3)/((0.3-e(1,k-1))*(-.3-e(1,k-1))))*safety;
    BF2 = log((-0.3*0.3)/((0.3-e(1,k-1))*(-.3-e(1,k-1))))*safety;

    %      end
    %          BF = log((-.2*0.2)/((0.2-e(1,k-1))*(-0.2-e(1,k-1))))*0
    %     end
    %BF = log(1*(-1-e(1,k-1))/(1*(1-e(1,k-1))))
    %BF = .8^2/pi*tan(pi*e(1,k-1)^2/2/.8^2)
    Yk_1 = e(:,k-1)'*Q*e(:,k-1)+u(:,k-1)'*R*u(:,k-1)- gamma2*w(:,k-1)'*w(:,k-1)+1000*BF1 +10000*BF2;
    delta_x = gamma*Critic_NL_gamma_bah(AugX(:,k))-Critic_NL_gamma_bah(AugX(:,k-1));
    EJK = Yk_1+W_critic(:,:,k)'*delta_x;
    YK = X(:,k-1)'*QN*X(:,k-1)+u(:,k-1)'*R*u(:,k-1);

    %%

    YY=Yk_1;
    phix = gamma*Critic_NL_gamma_bah(AugX(:,k))-Critic_NL_gamma_bah(AugX(:,k-1));

    if expReplay == true && k>1
        if k<=S
            Q11=[Q11,phix];
            Q22=[Q22,YY];
            Q33=[Q33,u(:,k)];
            Q44=[Q44,phix./(phix'*phix+1)^2];
        end

        ee=(Q22+W_critic(:,k)'*Q11);

        P1=Q44*ee';


        if k>S
            vv=0;
            hh=norm(phix);
            for ll=1:S-1

                m11=norm(Q11(:,ll));
                m1(ll)=m11;
                if hh>m1(ll)
                    vv=vv+1;
                end
            end

            if vv>=1
                [i1,j1]=min(m1);
                Q11(:,j1)=phix';
                Q22(:,j1)=YY;
                Q33(:,j1)=u(:,k);
                Q44(:,j1)=phix./(phix'*phix+1)^2;
            end
        end
    end
    %%

    temp_1 = W_critic(:,:,k);


    Jhat(k) = W_critic(:,k)'*Critic_NL_gamma_bah(AugX(:,k));
    Jhatsum(k+1)=Jhatsum(k)+Jhat(k);




    for LL=1:L
        delta_xK = gamma*Critic_NL_gamma_bah(AugX(:,k))-Critic_NL_gamma_bah(AugX(:,k-1));
        XK = delta_xK;

        if LL ==1
            if 3000 <= k && k < 6000

                temp_1 =temp_1-aj*(delta_xK*EJK)/(delta_xK'*delta_xK+1)-PE_C*aj*M1*P1-1*aj*(temp_1-W_critic(:,:,2999))*EWC;
            elseif 6000<=k
                temp_1 =temp_1-aj*(delta_xK*EJK)/(delta_xK'*delta_xK+1)-PE_C*aj*M1*P1-1*aj*(temp_1-W_critic(:,:,5999))*EWC;
            else
                temp_1 =temp_1-aj*(delta_xK*EJK)/(delta_xK'*delta_xK+1)-PE_C*aj*M1*P1;
            end


        else
            temp_1 =temp_1-aj*(delta_xK*EJK)/(delta_xK'*delta_xK+1);

        end
    end
    W_critic(:,:,k+1)=temp_1;

    if k>201
        disturbance(k)=0.1*exp(-0.1*(k-200)*1)*dist_on;
        %         disturbance(k)=0.05*sin(-0.1*(k-200)*1);
    end


    [e(:,k+1),r(:,k+1),Gx] = NLinear_sys_NL_gamma_bah(x(:,k),u(:,k),disturbance(k),H,r(:,k),k);
    x(:,k+1)=e(:,k+1)+r(:,k+1);
    X(:,k+1) = [e(:,k+1);r(:,k+1);1];
    AugX(:,k+1) = [e(:,k+1);r(:,k+1)];
    %      u_tilda = W_actor(:,:,k)'*Actor_NL_gamma_bah(X(:,k),v_actor(:,:,k),Neuron_Num_a)+gamma*0.5*inv(R)*Gx'*Dsigdx_NL_gamma_bah(X(:,k+1),v_critic(:,:,k),Neuron_Num_c)'*W_critic(:,:,k);

    u_tilda = W_actor(:,:,k)'*Actor_NL_gamma_bah(X(:,k),v_actor(:,:,1),Neuron_Num_a)+gamma*0.5*inv(R)*Gx'*Dsigdx_NL_gamma_bah(AugX(:,k+1))'*W_critic(:,:,k+1);

    temp = Actor_NL_gamma_bah(X(:,k),v_actor(:,:,1),Neuron_Num_a)/(Actor_NL_gamma_bah(X(:,k),v_actor(:,:,1),Neuron_Num_a)'*Actor_NL_gamma_bah(X(:,k),v_actor(:,:,1),Neuron_Num_a)+1);


    W_actor(:,:,k+1) = W_actor(:,:,k)-au*temp*u_tilda';

    if k<200
        u(:,k+1) = W_actor(:,:,k+1)'*Actor_NL_gamma_bah(X(:,k+1),v_actor(:,:,1),Neuron_Num_a)+0*rand;
        % u(:,k)=-[100,0,20,0;0,100,0,20]*e(:,k);
        % u(:,k+1)=-[20,0,100,0;0,20,0,100]*e(:,k);
    else
        u(:,k+1) = W_actor(:,:,k+1)'*Actor_NL_gamma_bah(X(:,k+1),v_actor(:,:,1),Neuron_Num_a);
    end

    wopt(:,k)=inv(gamma2)*0.5*H'*Dsigdx_NL_gamma_bah(AugX(:,k+1))'*W_critic(:,:,k+1);
    w_tilda = W_actor2(:,:,k)'*Actor_NL_gamma_bah(X(:,k+1),v_actor2(:,:,1),Neuron_Num_a2) - wopt(:,k);
    tempw = Actor_NL_gamma_bah(X(:,k+1),v_actor2(:,:,1),Neuron_Num_a2);
    W_actor2(:,:,k+1) = W_actor2(:,:,k)-aw*tempw*w_tilda';
    w(:,k+1) = W_actor2(:,:,k+1)'*Actor_NL_gamma_bah(X(:,k+1),v_actor2(:,:,1),Neuron_Num_a2);

    % u(:,k+1)=-[100,0,20,0;0,100,0,20]*e(:,k+1);
    %  u(:,k+1) = gamma*0.5*inv(R)*Gx'*Dsigdx_NL_gamma_bah(X(:,k),v_critic(:,:,k),Neuron_Num_c)'*W_critic(:,:,k);
    U_tilda(:,k)=u_tilda;
    Err(k)=EJK;
    %     cost(k)=YK;
    x(:,k+1)=e(:,k+1)+r(:,k+1);

end

figure(1);hold on;
% nn = 1:iter+1;
subplot 331; plot(x(1,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(r(1,:),'--r','LineWidth',2);
ylabel('x1,r1','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
% title('x1 and r1','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 332; plot(x(2,:),'b','LineWidth',2);
grid on;box on;
hold on
plot(r(2,:),'--r','LineWidth',2);
ylabel('x2,r2','FontWeight','b','FontSize',12);
xlabel('Iteration','FontWeight','b','FontSize',12);
% title('x2 and r2 ','FontWeight','b','FontSize',12);
set( gca, 'FontWeight', 'b','FontSize', 12 );





subplot 333; plot(e(1,:),'--r','LineWidth',2);hold on;plot(e(2,:),'--b','LineWidth',2);hold on;
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('Tracking error','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

% subplot 334;
% for j=1:m
%     for i = 1:Neuron_Num_c
%         hold on;
%         P(:)=W_critic(i,1,:);
%         plot(P,'LineWidth',2);
%     end
% end
% xlabel('Sampling instants','FontWeight','b','FontSize',12);
% ylabel('W_{critic}','FontWeight','b','FontSize',12);
% %  title('Weights variation','FontWeight','b','FontSize',12);
% grid on;box on;
% set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 335; plot(Err,'b');hold on;
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('TDE','FontWeight','b','FontSize',12);
% title('Error','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 336;
for i =1:m
    plot(U_tilda(i,:),'r');
end
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('$\tilde{u}$','FontWeight','b','FontSize',16,'Interpreter','latex');
% title('Utilda','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

subplot 337;
for i=1:m
    plot(u(i,:),'r');
end
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('$\hat{u}$','FontWeight','b','FontSize',16,'Interpreter','latex');
% title('\hat{u}','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );
%
% subplot 338;
% plot(J_sum,'r');
% xlabel('Sampling instants','FontWeight','b','FontSize',12);
% ylabel('Actual Cost','FontWeight','b','FontSize',12);
% % title('Estimated Cost','FontWeight','b','FontSize',12);
% grid on;box on;
% set( gca, 'FontWeight', 'b','FontSize', 12 );
%
subplot 334;
plot(Jhatsum,'k');
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('Estimated Value Function','FontWeight','b','FontSize',12);
% title('Estimated Cost','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

set(gcf,'units','normalized','outerposition',[0 0 1 1])

% Weights plot
figure;
subplot 221;
for i = 1:Neuron_Num_c
    hold on;
    plot(W_critic(i,:),'LineWidth',2);
end
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('W_{critic}','FontWeight','b','FontSize',12);
%  title('Weights variation','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

% for i = 1:length(v_actor)
%
%     temp_m2 = v_actor(:,:,i);
%     Vactor(:,i) = temp_m2(:);
% end



subplot 223;
for j=1:m
    for i = 1:Neuron_Num_a
        hold on;
        P(:)=W_actor(i,j,:);
        plot(P,'LineWidth',2);
    end
end
xlabel('Sampling instants','FontWeight','b','FontSize',12);
ylabel('W_{actor}','FontWeight','b','FontSize',12);
%  title('Weights variation','FontWeight','b','FontSize',12);
grid on;box on;
set( gca, 'FontWeight', 'b','FontSize', 12 );

% subplot 224;
% for i = 1:Neuron_Num_c
%     hold on;
%     plot(Vactor(i,:),'LineWidth',2);
% end
% xlabel('Sampling instants','FontWeight','b','FontSize',12);
% ylabel('V_{actor}','FontWeight','b','FontSize',12);
% %  title('Weights variation','FontWeight','b','FontSize',12);
% grid on;box on;
% set( gca, 'FontWeight', 'b','FontSize', 12 );


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
