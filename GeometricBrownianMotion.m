% Script (I) Imports data from .csv file and calculates volatility & mean.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all,
clc
OmxSeCP=readtable('OmxSe.csv').AdjClose; % OMXS30, 5th oct 2021 to 5th oct 2022.
N=size(OmxSeCP);
n=N(1,1);
dt=1/252;
logRet=diff(log(OmxSeCP));
meanO=mean(logRet); % mu
S=zeros(n,1);
for i=1:n-1
    S(i,1)= (logRet(i,1)-meanO)^2;
end
sumO=sum(S);
S2u=1/(n-1)*sumO;
sigma=S2u/sqrt(dt) % estimated volatility
meanO
% Output: sigma(daily)= 0.002846212151597, mean(daily)= —0.000697629857478299
% Annualized: sigma(yr)=0.04518, mean(yr)= -0.1758




% Script (II) Produces a series of stock price trajectories over a year with
% Geometric Brownian Motion.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all,
clc
S0=1909.6; % Choose initial stock price value
n=252; % Choose number of how many desired data points
mu=0.05; % Here, choose drift-coefficient value (yearly average).
sigma=0.30; % Here, choose yearly volatility value.
max=3; % Desired # of GBM plots (run code again to gen. more trajectories)
h=10/n; % Making the X-vector for plot
sh=sqrt(h);
r=mu-sigma^2/2;
mh=r*h; % r has mu & sigma. mh is like the multiplier to priceinc
for nrgraphs=1:max
    priceincrement=S0;
    P(:,1)=[0 S0];
    % For putting prices on n=252 places (setting y values to x-axis)
    for incrementx=2:n % idea is graph either moving a bit up or down
        random=randn(); % randomness into price movement
        priceincrement=priceincrement*exp(mh+sh*sigma*random);
        P(:,incrementx)=[incrementx*h priceincrement];
    end
    plot(P(1,:),P(2,:));
    hold on;
end
title('Script (II): Simulated GBM with drift (5%) & volatility (30%)');
xlabel('time: 1 year');
ylabel('OMXS30, simulated price value');




% Script (III) Produces a histogram of a year based on Geometric Brownian motion,
% put & call evaluations and plots option price convergence.
% With N larger than 50,000 this script takes a long time to finish.
% Finally, the script calculates Asian option price.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc
dt=1/252; % Time Step Delta_t (dt)
S0=1909.6; % Initial Stock Price S0
mu=0.05; % Drift mu
Sigma=0.3; % Volatility Sigma
n=252; % Interval size for Asian option
Time=linspace(dt,0,1);
S=ones(n,1);
Drift=ones(n,1);
Uncertainty=ones(n,1);
r=zeros(n,1);
% dSt= St*mu*dt+ St*sigma*dz / Drift + Uncertainty
Drift(1,1)=mu*dt*S0;
Uncertainty(1,1)=normrnd(0,1)*sqrt(dt)*Sigma*S0;% dz= epsilon*sqrt(dt)
change(1,1)=Drift(1,1)+Uncertainty(1,1);
N=50000;
St=zeros(N,1);
alpha=mu-Sigma^2/2;
K=1950;
E_PayoffEuC=0;
E_PayoffEuP=0;
E_PayoffAsC=0;
E_PayoffAsP=0;
for j=1:N
S(1,1)=S0;
for i=2:n
S(i,1)=S(i-1,1)*exp((alpha)*dt+normrnd(0,1)*sqrt(dt)*Sigma);
end
St(j,1)=S(end,1);
E_PayoffEuC=E_PayoffEuC+max((St(j,1)-K),0)*1/N; % for EU Call Option
E_PayoffEuP=E_PayoffEuP+max(K-(St(j,1)),0)*1/N; % for EU Put Option
E_EuCdisc=E_PayoffEuC*exp(-mu); % Discounted to time 0 (Call Premium)
E_EuPdisc=E_PayoffEuP*exp(-mu); % Discounted to time 0 (Put Premium)
meanS(j,1)=mean(S); % Simplified with 1 sample data, interval size n.
E_PayoffAsC=E_PayoffAsC+ max((meanS(j,1)-K),0)*1/N; % for Asian Call Option
E_PayoffAsP= E_PayoffAsP+ max((K-meanS(j,1)),0)*1/N; % for Asian Put Option
end
