
function [output1,output2,Gx,Zout] = NLinear_sys_NL_gamma_bah(x,u,w,H,r,t,Zin,E)
%----------------------Linear system Dynamics-------------------------------
T=0.01;
x1=x(1);x2=x(2);
F11=x1+T.*x2;
F22=x2+T*(-x1 + .1*(1-x1^2)*x2);
F = [F11;F22];


G = T*[0;1];


if t<=3000 

   rs = 1*exp(-t/4*T)*[sin((t)*T) , cos((t)*T)-1/4*sin((t)*T)]';
 
elseif 6000<t
    t=t-6000;
     rs = 1*exp(-t/4*T)*[sin((t)*T) , cos((t)*T)-1/4*sin((t)*T)]';
else 
    t=t-3000;
    rs = 1*exp(-t/4*T)*[sin(0.5*(t)*T) , 0.5*cos(0.5*(t)*T)-1/4*sin(0.5*(t)*T)]';
end
 
Gx=[G;zeros(2,1);zeros(2,1)];
 Fe=F-rs;

xk1 = Fe+G*u+H(1:2)*w;

output1 = xk1; 

output2 = rs;

Zout=Zin-T*E;
end

