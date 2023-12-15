function result = gwrlrp_aicc(y,x,east,north,info)
% PURPOSE: compute Linear Response with Plateau Geographically Weighted Regression
%----------------------------------------------------
% USAGE: results = gwr(y,x,east,north,info)
% where:   y = dependent variable vector
%          x = explanatory variable matrix
%       east = x-coordinates in space
%      north = y-coordinates in space
%       info = a structure variable with fields:
%       info.bwidth = scalar bandwidth to use or zero
%                     for cross-validation estimation (default)
%       info.bmin   = minimum bandwidth to use in CV search
%       info.bmax   = maximum bandwidth to use in CV search
%                     defaults: bmin = 0.1, bmax = 20
%       info.dtype  = 'gaussian'    for Gaussian weighting (default)
%                   = 'exponential' for exponential weighting
%                   = 'tricube'     for tri-cube weighting
%       info.q      = q-nearest neighbors to use for tri-cube weights
%                     (default: CV estimated)
%       info.qmin   = minimum # of neighbors to use in CV search
%       info.qmax   = maximum # of neighbors to use in CV search
%                     defaults: qmin = nvar+2, qmax = 4*nvar
% ---------------------------------------------------
%  NOTE: res = gwr(y,x,east,north) does CV estimation of bandwidth
% ---------------------------------------------------
% RETURNS: a results structure
%        results.meth  = 'gwr'
%        results.beta  = bhat matrix    (nobs x nvar)
%        results.tstat = t-stats matrix (nobs x nvar)
%        results.yhat  = yhat
%        results.resid = residuals
%        results.sige  = e'e/(n-dof) (nobs x 1)
%        results.nobs  = nobs
%        results.nvar  = nvars
%        results.bwidth  = bandwidth if gaussian or exponential
%        results.q       = q nearest neighbors if tri-cube
%        results.dtype   = input string for Gaussian, exponential weights
%        results.iter    = # of simplex iterations for cv
%        results.north = north (y-coordinates)
%        results.east  = east  (x-coordinates)
%        results.y     = y data vector
%---------------------------------------------------
% See also: prt,plt, prt_gwr, plt_gwr to print and plot results
%---------------------------------------------------
% References: Brunsdon, Fotheringham, Charlton (1996)
% Geographical Analysis, pp. 281-298
%---------------------------------------------------
% NOTES: uses auxiliary function scoref for cross-validation
%---------------------------------------------------

% written by: James P. LeSage 2/98
% University of Toledo
% Department of Economics
% Toledo, OH 43606
% jpl@jpl.econ.utoledo.edu
%
% Modified by: Dayton M. Lambert7/23/2020
% Oklahoma State University
% Department of Agricultrual Economics
% Stillwater, Ok 74078
% dayton.lambert@okstate.edu
%
% Revised by Dayton M. Lambert 08-03-2020
% AICc cross validation with Hurvich's AICc
% Code modified for Linear Response & Plateau function

if nargin == 5 % user options
    if ~isstruct(info)
        error('gwr: must supply the option argument as a structure variable');
    else
        fields = fieldnames(info);
        nf = length(fields);
        % set defaults
        [~, k] = size(x);
        bwidth = 0; dtype = 0; q = 0; qmin = k+2; qmax = 5*k;
        bmin = 0.1; bmax = 20.0;
        for i=1:nf
            if strcmp(fields{i},'bwidth')
                bwidth = info.bwidth;
            elseif strcmp(fields{i},'dtype')
                dstring = info.dtype;
                if strcmp(dstring,'gaussian')
                    dtype = 0;
                elseif strcmp(dstring,'exponential')
                    dtype = 1;
                elseif strcmp(dstring,'bisquare')
                    dtype = 2;
                end
            elseif strcmp(fields{i},'q')
                q = info.q;
            elseif strcmp(fields{i},'qmax')
                qmax = info.qmax;
            elseif strcmp(fields{i},'qmin')
                qmin = info.qmin;
            elseif strcmp(fields{i},'bmin')
                bmin = info.bmin;
            elseif strcmp(fields{i},'bmax')
                bmax = info.bmax;
            end
        end % end of for i
    end % end of if else
elseif nargin == 4
    bwidth = 0; dtype = 0; dstring = 'gaussian';
    bmin = 0.1; bmax = 20.0;
else
    error('Wrong # of arguments to gwr');
end

% error checking on inputs
[nobs, nvar] = size(x);
[nobs2 , ~] = size(y);
[nobs3 , ~] = size(north);
[nobs4 , ~] = size(east);

result.north = north;
result.east = east;

if nobs ~= nobs2
    error('gwr: y and x must contain same # obs');
elseif nobs3 ~= nobs
    error('gwr: north coordinates must equal # obs');
elseif nobs3 ~= nobs4
    error('gwr: east coordinates must equal # in north');
end

switch dtype
    case{0,1} % bandwidth cross-validation
        if bwidth == 0 % cross-validation
            options = optimset('fminbnd');
            optimset('MaxIter',5000);
            if dtype == 0     % Gaussian weights
                [bdwt,~,~,output] = fminbnd('scoreflrp_aicc',bmin,bmax,options,y,x,east,north,dtype,info.beta0,info.J0);
            elseif dtype == 1 % exponential weights
                [bdwt,~,~,output] = fminbnd('scoreflrp_aicc',bmin,bmax,options,y,x,east,north,dtype,info.beta0,info.J0);
            end
            if output.iterations == 5000
                fprintf(1,'gwr: cv convergence not obtained in %4d iterations',output.iterations);
            else
                result.iter = output.iterations;
            end
        else
            bdwt = bwidth*bwidth; % user supplied bandwidth
        end
    case{2} % q-nearest neigbhor cross-validation
        if q == 0 % cross-validation
            q = scoreqlrp_aicc(qmin,qmax,y,x,east,north,info.beta0,info.J0);
        else
            % use user-supplied q-value
        end
    otherwise
end

% do GWR using bdwt as bandwidth
[n, k] = size(x);
bsave = zeros(n,k+1);
ssave = zeros(n,k+1);
sigv  = zeros(n,1);
yhat  = zeros(n,1);
resid = zeros(n,1);
wt = zeros(n,1);

lb = [0,0,0];
ub = [max(y),100,max(y)];

yhat0 = min(info.beta0(1,1)  + info.beta0(2,1)*x(:,2), info.beta0(3,1) );

ehat0 = y - yhat0;
RSS0 = ehat0'*ehat0;
sighat0 = RSS0/(nobs - cols(info.beta0) );

result.RSS0 = RSS0;

L = zeros(nobs,nobs);
B = zeros( nobs,nobs,cols(bsave)  );

hwait = waitbar(0,'Running GWR...');

J0 = info.J0;

xpx=J0'*J0;
ixpx=xpx\eye(cols(J0 ));

Q = J0*ixpx*J0';
IQ=speye(n) - Q;

IQPIQ = IQ'*IQ;
del0_1 = trace(IQPIQ);
del0_2= trace(IQPIQ^2);
nu0_1 = trace(Q'*Q);
result.nu0_1=  nu0_1;
result.AICc0 = nobs*log(RSS0/nobs)+nobs*((del0_1/del0_2*(nobs + nu0_1))/(((del0_1^2)/del0_2)-2));

for iter=1:n
    dx = east - east(iter,1);
    dy = north - north(iter,1);
    d = sqrt(dx.*dx + dy.*dy);
    sd = std(d);
    % sort distance to find q nearest neighbors
    ds = sort(d);
    if dtype == 2, dmax = ds(q,1); end
    if dtype == 0     % Gausian weights
        wt = stdn_pdf(bdwt*d/sd);
    elseif dtype == 1 % exponential weights
        wt = exp(-d*bdwt/sd);
    elseif dtype == 2 % tricube weights
        wt = zeros(n,1);
        nzip = find(d <= dmax);
        wt(:,1) = (d(:,1) <= dmax ).*(1-(d(:,1)/dmax).^2).^3;
 
    end % end of if,else
    wt = sqrt(wt);
    
    % computational trick to speed things up
    % use non-zero wt to pull out y,x observations
    nzip = find(wt >= 0);
    ys = y(nzip,1).*wt(nzip,1);
    xs = matmul(x(nzip,:),wt(nzip,1));
    fun = @(beta)(min(beta(1) + beta(2)*x(:,2), beta(3) ) -  y(:,1)).*wt(:,1)  ;
    options=  optimset('display','off' );
    [beta, ~, ~, ~, ~, ~, ~ ] = lsqnonlin(fun,info.beta0, lb, ub, options );
    W = diag(wt.^2);
    XPX =  J0'*W*J0;
    JW = (J0'*W)';
    iXPX = (XPX + 1e-6*eye(cols(bsave)) )\eye(cols(bsave));
    B_ = iXPX*JW';
    L(iter, :) = J0(iter,:)*B_;
    
    for j=1:cols(bsave)
        if j==1,            c = [ 1 , zeros(1,cols(bsave)-1 ) ];
        end
        if (1< j) && (j <cols(bsave)), c = [zeros(1, j-1 ), 1, zeros(1,cols(bsave) - j)  ];
        end
        if j==cols(bsave),  c = [zeros(1,cols(bsave) -1), 1  ];
        end
        B(iter,:,j ) = c*B_ ;
    end
    
    % compute predicted values
    yhatv = min(beta(1)*xs(:,1) + beta(2)*xs(:,2), beta(3) );
    
    yhat(iter,1) = min(beta(1) + beta(2)*x(iter,2), beta(3)) ;
    resid(iter,1) = y(iter,1) - yhat(iter,1);
    % compute residuals
    e = ys - yhatv;
    % find # of non-zero observations
    nadj = length(nzip);
    sige = (e'*e)/nadj;
    
    % compute t-statistics
    sdb = sqrt(sige*diag(iXPX));
    % store coefficient estimates and std errors in matrices
    % one set of beta,std for each observation
    bsave(iter,:) = beta;
    ssave(iter,:) = sdb';
    sigv(iter,1) = sige;
    waitbar(iter/nobs);
end

close(hwait);

Jn = ones(nobs, nobs);
In=speye(nobs);
IL = In - L;
M1 = IL'*IL;
V0 = resid'*resid;
result.RSSg = V0;

delta1 = trace(M1);
delta2 = trace(M1^2);

result.delta1=delta1;
result.delta2=delta2;

gamma1 = zeros(cols(bsave),1);
gamma2 = zeros(cols(bsave),1);

A = (In - (1/nobs)*Jn);

A1 = IQ - M1;
A2 = IQ - 2*M1 + M1^2;
v1=trace(A1);
v2=trace(A2);

for k=1:cols(bsave)
    gamma1(k) =  trace(B(:,:,k)'*A*B(:,:,k)/nobs) ;
    gamma2(k) = trace((B(:,:,k)'*A*B(:,:,k)/nobs)^2) ;
end

% gamma2=gamma1.^2;
result.gamma1=gamma1;
result.gamma2 = gamma2;

% global f-test
result.sig2hat = V0/delta1;
result.F2 = ((RSS0 - V0)/(nobs - cols(bsave) - delta1))/sighat0;
result.F1 = result.sig2hat/sighat0 ;
result.denomdf = nobs-cols(bsave);

result.F1Pval = fcdf(result.F1,(delta1^2)/delta2,result.denomdf );
result.F2Pval = fcdf(result.F1,(v1^2)/v2,result.denomdf );
result.F1Crit = finv(0.05,(delta1^2)/delta2,result.denomdf ); 

% parameter stationarity test
result.Fstats  = zeros(cols(bsave),1);
result.pvalF   = zeros(cols(bsave),1);

Vk=var(bsave,1);
 
for k =1:cols(bsave)
    result.Fstats(k) = (Vk(k)/(gamma1(k) ))/result.sig2hat ;
    result.pvalF(k) =   fcdf(result.Fstats(k),(gamma1(k)^2 )/(gamma2(k) ),(delta1^2)/delta2,'upper');
end

result.gamma1=gamma1;

% fill-in results structure
result.meth = 'gwr';
result.nobs = nobs;
result.nvar = nvar;
if (dtype == 0 || dtype == 1)
    result.bwidth = sqrt(bdwt);
else
    result.q = q;
end
result.beta = bsave;
%result.tstat = bsave./ssave;
result.sige = sigv;
result.dtype = dstring;
result.y = y;
result.Vk=Vk;
result.yhat = yhat;
% compute residuals and conventional r-squared
result.resid = resid;
sigu = result.resid'*result.resid;
ym = y - mean(y);
rsqr1 = sigu;
rsqr2 = ym'*ym;
result.rsqr = 1.0 - rsqr1/rsqr2; % r-squared
rsqr1 = rsqr1/(nobs-nvar);
rsqr2 = rsqr2/(nobs-1.0);
result.rbar = 1 - (rsqr1/rsqr2); % rbar-squared
result.v1=v1;
result.v2=v2;
nu1=trace(L'*L);
trH = trace(L);
%result.aicc= nobs*log(result.RSSg/nobs) +  nobs*((delta1/delta2*(nobs + nu1))/(((delta1^2)/delta2)-2));

result.aicc=  log(result.RSSg/nobs ) + 1 + (2*trH + 1)/(nobs-trH-2);