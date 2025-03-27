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
figure(1)
histogram(St); hold on
histfit(St)
format bank
mean(St)
% Example Output: mean(St)=2003.48, E_EuCdisc=252.97, E_EuPdisc=197.64
% Example Output 2 (Asian Option): E_PayoffAsP= 130.2821, E_PayoffAsC= 139.9824
title('Script (III): Simulated GBM with drift (5%) & volatility (30%)');
xlabel('Price intervals');
ylabel('Number of ST per interval');
%figure(2)
%plot(Time,S,'b'); % This command plots one of the many random walks.




%% Script (III)s option price convergence plot (Run sections separately)
T=1;
N=linspace(100,1000000,10000); % number of sims
for i=1:10000
    St = S0 * exp((alpha) * T + Sigma*sqrt(T) * normrnd(0, 1, [1, N(1,i)]));
    payoffEU_C(i,1) = exp(-mu * T) * mean(max(St - K, 0));
    payoffEU_P(i,1) = exp(-mu * T) * mean(max(K - St, 0));
end
figure(1)
plot(payoffEU_C); hold on
plot(payoffEU_P); hold off
title('Script (III)s option price convergence plot');
xlabel('Simulations');
ylabel('Option price');




% Script (IV) simulates Geometric Brownian Motion at time 1 immediately
% and determines expected option payoff and discounts it to time zero.
% It concludes by plotting option price convergence.
% The script can not plot GBM trajectories.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 1:
clear all,
clc
%
%
S0 = 1909.6; % initial price
T = 1; % time to maturity
mu = 0.05; % short rate
Sigma = 0.3; % volatility
K = 1950; % strike price
%N = 1000; % number of sims
%N = 10000; % number of sims
%N = 100000; % number of sims
N = 1000000; % number of sims
alpha = mu - (Sigma^2)/2;
St = S0 * exp((alpha) * T + Sigma*sqrt(T) * normrnd(0, 1, [1, N]));
E_PayoffEuC = exp(-mu * T) * mean(max(St - K, 0));
E_PayoffEuP = exp(-mu * T) * mean(max(K - St, 0));
E_EuCdisc = E_PayoffEuC*exp(-mu*T); % Discounted to time 0 (Call Premium)
E_EuPdisc = E_PayoffEuP*exp(-mu*T); % Discounted to time 0 (Put Premium)
mean(St);
% Example Output: Call=252.6112, Put=198.1543, mean(St)=2007.2489




%% Script (IV)s option price convergence plot (Run sections separately)
N=linspace(100,1000000,10000); % number of sims
for i=1:10000
    St = S0 * exp((alpha) * T + Sigma*sqrt(T) * normrnd(0, 1, [1, N(1,i)]));
    payoffEU_C(i,1) = exp(-mu * T) * mean(max(St - K, 0));
    payoffEU_P(i,1) = exp(-mu * T) * mean(max(K - St, 0));
end
plot(payoffEU_C); hold on
plot(payoffEU_P); hold off
% log(N)
title('Script (IV)s option price convergence plot');
xlabel('Simulations');
ylabel('Option price');




% Script (V) simulates Geometric Brownian Motion with MATLAB’s built in
% function over a year and plotts a histogram, which confirms Script (III).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc
dt=1/252;
t=252;
S0=1909.6;
mu=0.05;
sigma=0.3;
K=1950;
N=1000000;
GBM = gbm(mu, sigma, "StartState", S0);
[x, y] = GBM.simulate(t, "DeltaTime", dt, "Ntrials", N);
figure(1)
histogram(x(end,:,:),50); hold on
mean(x(end,:,:))
%figure(2) % For plotting N simulated price movements
%plot(y, squeeze(x))
title('Script (V): Simulated GBM with drift (5%) & volatility (30%)');
xlabel('Price intervals');
ylabel('Number of ST per interval');
% Example Output: mean(St)=2007.7




% Script(VI) is a basic Black-Scholes Model valuation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu=0.05;
sigma=0.3;
S0=1909.6;
K=1950;
[Call,Put] = blsprice(S0,K,mu,1,sigma)
% Definitive Output: Call=252.8264, Put=198.1238 (serves as benchmark)
% For S0=2831.2 from script (VII), and K=2900, Black Scholes results:
% Call=370.7918, Put=298.1821




% Script(VII) is a mixture of previous scripts. It simulates Geometric
% Brownian Motion and calculates European Basket option payoff and
% discounts time to zero. It then plots the option price convergence.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S0M = [1909.6 3752.75]; % initial price for OMX and S&P
weights = [.5; .5];
S0 = S0M * weights; % in this case, S0=2831.2
T = 1; % time to maturity
mu = 0.05; % short rate
sigma = 0.3; % volatility
K = 2900; % strike price
alpha = mu - (sigma^2)/2;
N = 1000000;
dN = 1000;
E_PayoffEuC = zeros(N/dN,1);
E_PayoffEuP = zeros(N/dN,1);
for n = dN:dN:N
    St = S0 * exp((mu - 0.5 * sigma^2) * T + sigma*sqrt(T) * normrnd(0, 1, [1,n]));
    E_PayoffEuC(n/dN,1) = exp(-mu * T) * mean(max(St - K, 0));
    E_PayoffEuP(n/dN,1) = exp(-mu * T) * mean(max(K - St, 0));
end
E_EuCdisc = E_PayoffEuC*exp(-mu*T); % Discounted to time 0 (Call Premium)
E_EuPdisc = E_PayoffEuP*exp(-mu*T); % Discounted to time 0 (Put Premium)
hold on
plot(E_PayoffEuC)
plot(E_PayoffEuP)
hold off
% Example Output: Call=370.7238, Put=298.1846