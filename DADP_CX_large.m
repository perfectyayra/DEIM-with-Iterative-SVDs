function [irow,icol,M,rounds_irow,rounds_col] = DADP_CX_large(A, k, delta, l)

%CX_DEIM_INOUT_DELTA_LARGE incurred CUR decomposition with SVD every iteration
% with one-sided projected residual; selects indices per round based on
% the decay pattern of the singular values
% for large-scale matrices
% function [irow, icol, M] = DADP_CX_large(A, k, delta, l)
% k = desired number of indices 
% delta = upper bound on the relative magnitudes of the singular values
% l = upper bound on the number of singular vectors to be computed per
% iteration
% C = A(:,icol);  R = A(irow,:)
% See also CUR_DEIM
%
% Revision date: June 29, 2023
% (C) Perfect Gidisu, Michiel Hochstenbach 2023

if nargin < 2 || isempty(k),     k     =   2; end
if nargin < 3 || isempty(delta), delta = 0.8; end
if nargin < 4 || isempty(l), l = 2; end
irow = []; icol = [];  Q = zeros(size(A,1),k);rounds_irow=[];rounds_col=[];
[opts.nr, opts.tol, opts.mindim, opts.maxdim, opts.v1] = deal(l, 1e-5/ sqrt(l), l, 2*l, ones(size(A,2),1)/sqrt(size(A,2)));


while length(icol) < k  
  if isempty(icol)
    [s, V, U] = krylov_schur_svd_relsigma(@(x,t) Bmv(x,t,A,[]), opts,delta); s1 = s;
  else  
    [s, V, ~] = krylov_schur_svd_relsigma(@(x,t) Bmv(x,t,A,Q(:,1:length(icol))), opts,delta);
    V(icol,:) = 0;
  end
  rounds_col=[rounds_col length(s)];
  tt = min([length(find(s >= delta*s(1))), k-length(icol),  l]);
  ic = deim(V(:,1:tt), tt);
  icol = [icol ic];
  ell = length(icol) - length(ic);
  Q1 = A(:,icol(ell+1:ell+tt));
  if ell > 0
    Q1 = Q1 - Q(:,1:ell)*(Q(:,1:ell)'*Q1); Q1 = Q1 - Q(:,1:ell)*(Q(:,1:ell)'*Q1); end
  [Q(:,ell+1:ell+tt), ~] = qr(Q1, 0);      
end

Q = zeros(size(A,2),k); opts.v1 = ones(size(A,1),1)/sqrt(size(A,1));
while length(irow) < k 
  if ~isempty(irow)
    [s1, U, ~] = krylov_schur_svd_relsigma(@(x,t) Bmv(x,t,A',Q(:,1:length(irow))), opts,delta);
    U(irow,:) = 0;
  end
  rounds_irow=[rounds_irow length(s1)];
  tt = min([length(find(s1 >= delta*s1(1))), k-length(irow),  l]);
  ir = deim(U(:,1:tt), tt);
  irow = [irow ir];
  ell = length(irow) - length(ir);
  Q1 = A(irow(ell+1:ell+tt),:)';
  if ell > 0
    Q1 = Q1 - Q(:,1:ell)*(Q(:,1:ell)'*Q1); Q1 = Q1 - Q(:,1:ell)*(Q(:,1:ell)'*Q1); end
    [Q(:,ell+1:ell+tt), ~] = qr(Q1, 0);  
end
M = A(:,icol) \ (A / A(irow,:));

function y = Bmv(x, t, A, Q)
if t == 0
  y = A*x; if ~isempty(Q), y = y - Q*(Q'*y); end
else
  if isempty(Q), y = A'*x; else, y = x - Q*(Q'*x); y = A'*y; end
end
