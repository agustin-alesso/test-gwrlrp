function q = scoreqlrp_aicc(qmin,qmax,y,x,east,north,bstart,J0)
% PURPOSE: evaluates cross-validation score for optimal q in gwr
%          based on tricube weighting
% ------------------------------------------------------
% USAGE: score = scoreq(qmin,qmax,y,x,east,north);
% where: qmin = minimum # nearest neighbors to use in CV search
%        qmax = maximum # nearest neighbors to use in CV search
%        y    = dependent variable
%        x = matrix of explanatory variables
%     east = longitude (x-direction) coordinates
%    north = lattitude (y-direction) coordinates
% ------------------------------------------------------
% RETURNS: q = # of nearest neighbors that minimum the score
%              function
% ------------------------------------------------------
% NOTE: this function catches inversion problems
%       and uses Hoerl-Kennard ridge regression
%       if needed
% ------------------------------------------------------
% SEE ALSO: scoref which finds optimal bandwidth
%           for gaussian and exponential weighting
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
% Code modified for Linear Response & Plateua function

[n , ~] = size(x); res = zeros(n,1);

qgrid = qmin:qmax;
nq = length(qgrid);
wt = zeros(n,nq);

options=  optimset('display','off');
L = zeros(n ,n );
nu1=zeros(1,nq);
delta1=nu1;
delta2=nu1;
lb = [0,0,0];
ub = [max(y),100,max(y)];
for iter = 1:n
    dx = east - east(iter,1);
    dy = north - north(iter,1);
    d = sqrt(dx.*dx + dy.*dy);
    % sort distance to find q nearest neighbors
    ds = sort(d);
    dmax = ds(qmin:qmax,1);
    for j=1:nq
        wt(:,j) = (d(:,1) <= dmax(j,1)).*(1-(d(:,1)/dmax(j,1)).^2);
        wt(iter,j) = 0.0;
    end % end of j loop
    for j=1:nq
        fun = @(beta)(min(beta(1) + beta(2)*x(:,2), beta(3) ) -  y(:,1)).*wt(:,j)  ;
        [beta, ~, ~, ~, ~, ~, ~ ] = lsqnonlin(fun,bstart, lb, ub, options );
        W =diag(wt(:,j).^2);
        XPX =  J0'*W*J0;
        iXPX = (XPX + 1e-6*eye(cols(x)+1) )\eye(cols(x)+1);
        JW = (J0'*W)';
        B_ = iXPX*JW';
        L(iter, :) = J0(iter,:)*B_;
        ILPIL=(speye(n) - L)'*(speye(n) - L);
        % compute predicted values
        yhat = min(beta(1) + beta(2)*x(iter,2), beta(3));
        nu1(1,j)=trace(L'*L);
        delta1(1,j)=trace(ILPIL);
        delta2(1,j)=trace(ILPIL^2);
        % compute residuals
        res(iter,j) = y(iter,1) - yhat;
    end % end of for j loop over q-values
end % end of for iter loop

tmp = res.*res;
sse = sum(tmp);
sig2=sse./n;
trH = trace(L );
aicc=  log(sig2) + 1 + (2*trH + 1)/(n -trH-2);
disp(aicc)
score = aicc;
[~, sind] = min(score);
q = qgrid(sind);



