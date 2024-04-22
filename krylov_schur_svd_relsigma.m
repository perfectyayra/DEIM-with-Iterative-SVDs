function [sigma, V, U] = krylov_schur_svd_relsigma(A, opts, delta)

%KRYLOV_SCHUR_SVD  Krylov-Schur method for the SVD
% function [sigma, V, U, hist, mvs] = krylov_schur_svd_relsigma(A, opts, delta)
%
% Opts can have the following fields:
%   nr           number of desired singular triplets               1   
%   v1           initial space (may be more-dim)                   randn(n,1)
%   tol          tolerance of the outer iteration                  1e-6
%   absrel       absolute or relative tolerance outer iteration    'rel'
%                  relative tolerance: ||A'U-VB'|| < tol * ||A||_1
%   mindim       minimum dimension of subspaces                    10
%   maxdim       maximum dimension of subspaces                    20
%   maxit        maximum number of outer iterations                1000
%   target       Inf | 0                                           Inf
%   info         info desired (0,1,2,...)                          0
%
% See also NORMEST, KRYLOV_ATA, KRYLOV_ATA_EXPAND, KRYLOV_SCHUR
%
% Reference: Stoll, A Krylov–Schur approach to the truncated SVD, 2012
%
% Revision date: May 13, 2022
% (C) Michiel Hochstenbach 2022

if isnumeric(A)
  [m,n] = size(A);
else  % If A is function, a starting vector should be given
  n = length(opts.v1); m = length(mv(A,opts.v1,0));
end
if nargin < 2, opts = []; end

if isfield(opts, 'nr'),     nr        = opts.nr;     else nr     =        1; end
if isfield(opts, 'target'), target    = opts.target; else target =      Inf; end
if isfield(opts, 'v1'),     v1        = opts.v1;     else v1     = rand1(n); end
if isfield(opts, 'mindim'), m1        = opts.mindim; else m1     =       10; end
if isfield(opts, 'maxdim'), m2        = opts.maxdim; else m2     =       20; end
if isfield(opts, 'maxit'),  maxit     = opts.maxit;  else maxit  =     1000; end
if isfield(opts, 'tol'),    tol       = opts.tol;    else tol    =     1e-6; end
if isfield(opts, 'absrel'), absrel    = opts.absrel; else absrel =    'rel'; end
if isfield(opts, 'info'),   info      = opts.info;   else info   =        0; end
if strcmp(absrel, 'rel') && isnumeric(A), tol = tol * norm(A,1); end
if m1 < nr,   m1 = nr;   end
if m2 < 2*m1, m2 = 2*m1; end

if info
  fprintf('\n*** Krylov-Schur for the SVD ***\n\n');
  fprintf('  Size of problem:   %d x %d\n', m, n);
  fprintf('  Number of sigma''s: %d\n', nr);
  fprintf('  Target:            %g\n', target);
  fprintf('  Tolerance:         %g\n', tol);
  fprintf('  Dim search spaces: min %d, max %d\n', m1, m2);
  fprintf('  Max iterations:    %d\n\n', maxit);
  if info > 1, fprintf(' Iter  error     sigma\n'); fprintf('-------------------------\n'); end
end

B = zeros(m2, m2+1);
[V, U, alpha, beta] = krylov_ata(A, v1, m1);
B(1:m1+1, 1:m1+1) = diag([alpha 0]) + diag(beta, 1);

for k = 1:maxit
  [V, U, alpha, beta] = krylov_ata_expand(A, V, U, B(1:m1,m1+1), m2-m1); % Expand
  B(m1+1:m2, m1+1:m2) = diag(alpha) + diag(beta(1:m2-m1-1), 1);
  B(m2,m2+1) = beta(m2-m1);
  [X, S, Y] = svd(B(1:m2,1:m2));  sigma = diag(S);  % Extract
  if target == 0, sigma = sigma(m2:-1:1); X = X(:,m2:-1:1); Y = Y(:,m2:-1:1); end
  V = [element(V(:,1:m2)*Y, 1:n, 1:m1) V(:,m2+1)];  % Restart
  U = element(U(:,1:m2)*X, 1:m, 1:m1);
  e = element(B(:,m2+1)'*X, 1:m1);
  B(1:m1,1:m1+1) = [diag(sigma(1:m1)) e'];
  err = norm(e(1:nr));
  if nargout > 3, hist(k) = err; end
  if (info == 2) || (info > 1 && ~mod(k, info))
    fprintf('%4d  %6.2e', k, err);
    fprintf('  %6.2g', sigma(1:min(3,nr))); fprintf('\n')
  end
  i = length(find(e < tol));
  if i==nr || (i>0 && sigma(i) < delta*sigma(1))  % Convergence for first i
    sigma = sigma(1:i); V = V(:,1:i); U = U(:,1:i);
    if nargout > 4, mvs = (1:k)*(m2-m1)+m1; end
    if info, fprintf('Found after %d iterations with residual = %6.2e\n', k, err); end
    return
  end
end
if nargout > 4, mvs = 2*((1:k)*(m2-m1)+m1); end
if info, fprintf('Quit after max %d iterations with residual = %6.2e\n', k, err); end


function [V, U, alpha, beta] = krylov_ata_expand(A, V, U, c, k)
%KRYLOV_ATA_EXPAND  Expand orthonormal bases for Lanczos bidiagonalization spaces K_k(A'A, v) and K_k(AA', Av)
% function [V, U, alpha, beta] = krylov_ata_expand(A, V, U, c, k)
m = size(V,2);
V = [V zeros(size(V,1), k)];          % Preallocation
U = [U zeros(size(U,1), k)];
alpha = zeros(1,k); beta = zeros(1,k);
for j = m:(k+m-1)
  r = mv(A, V(:,j), 0);
  if j == m
    r = r - U(:,1:j-1)*c;
  else
    r = r - beta(j-m)*U(:,j-1);
  end
  r = r - U(:,1:j-1)*(U(:,1:j-1)'*r); % Reorthogonalization
  alpha(j-m+1) = norm(r);
  U(:,j) = r / alpha(j-m+1);
  r = mv(A, U(:,j), 1);
  r = r - alpha(j-m+1)*V(:,j);
  r = r - V(:,1:j)*(V(:,1:j)'*r);     % Reorthogonalization
  beta(j-m+1) = norm(r);
  V(:,j+1) = r / beta(j-m+1);
end
