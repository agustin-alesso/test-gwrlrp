function score = scoreflrp_aicc(bdwt,y,x,east,north,flag,bstart,J0)
% PURPOSE: evaluates cross-validation score for optimal gwr bandwidth
%          with gauusian or exponential weighting
% ------------------------------------------------------
% USAGE: score = scoref(y,x,east,north,bdwt);
% where: y = dependent variable
%        x = matrix of explanatory variables
%     east = longitude (x-direction) coordinates
%    north = lattitude (y-direction) coordinates
%     bdwt = a bandwidth to use in computing the score
%     flag = 0 for Gaussian weights
%          = 1 for BFG exponential
% ------------------------------------------------------
% RETURNS: score = a cross-validation criterion
% ------------------------------------------------------
% SEE ALSO: scoreq that determines optimal q-value for
%           tricube weighting
% ------------------------------------------------------

% written by: James P. LeSage 2/98
% University of Toledo
% Department of Economics
% Toledo, OH 43606
% jpl@jpl.econ.utoledo.edu
%
% Revised by Dayton M. Lambert 08-03-2020
% Oklahoma State University
% Department of Agricultrual Economics
% Stillwater, Ok 74078
% dayton.lambert@okstate.edu
% AICc cross validation with Hurvich's AICc
% Code modified for Linear Response & Plateau function


[n , ~] = size(x); res = zeros(n,1);
wt = zeros(n,1);
 
lb = [0,0,0];
ub = [max(y),100,max(y)];

options=  optimset('display','off');
L = zeros(n ,n );
for iter = 1:n
    dx = east - east(iter,1);
    dy = north - north(iter,1);
    d =  sqrt(dx.*dx + dy.*dy);
    sd = std(d);
    if flag == 0     % Gausian weights
        wt = stdn_pdf(bdwt*d/sd );
    elseif flag == 1 % exponential weights
        wt = exp(-d*bdwt/sd);
    end
    wt(iter,1) = 0.0;
    wt = sqrt(wt);
    
    % computational trick to speed things up
    % use non-zero wt to pull out y,x observations
    nzip = find(wt >=0);
    fun = @(beta)(min(beta(1) + beta(2)*x(nzip,2), beta(3) ) -  y(nzip,1)).*wt(nzip,1)  ;
    [beta, ~, ~, ~, ~, ~, ~ ]= lsqnonlin(fun,bstart, lb, ub, options );
    W=diag(wt.^2);
    XPX =  J0'*W*J0;
    iXPX = (XPX + 1e-6*eye(cols(x)+1) )\eye(cols(x)+1);
    JW = (J0'*W)';
    B_ = iXPX*JW';
    L(iter, :) = J0(iter,:)*B_;
    % compute predicted values
    yhat = min(beta(1) + beta(2)*x(iter,2), beta(3));
    % compute residuals
    res(iter,1) = y(iter,1) - yhat;
end % end of for iter loop
nu1=trace(L'*L);
ILPIL=(speye(n) - L)'*(speye(n) - L);
tmp = res'*res;
delta1=trace(ILPIL);
trH = trace(L );
aicc=  log(tmp/n) + 1 + (2*trH + 1)/(n -trH-2);
disp(aicc)
score = aicc;

