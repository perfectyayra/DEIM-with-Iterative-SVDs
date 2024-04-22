function [irow, icol, M] = CADP_CX_large(A, k, rounds)

%CX_DEIM_INOUT_LARGE incurred CUR decomposition with SVD every iteration with
% one-sided projected residual; selects constant number of indices per round
% for large-scale matrices
% function [irow, icol, M] = CADP_CX_large(A, k, rounds)
% k = desired number of indices 
% rounds = number of iterations, rounds | k
% C = A(:,icol);  R = A(irow,:)
%
% See also CUR_DEIM
%
% Revision date: June 29, 2023
% (C) Perfect Gidisu, Michiel Hochstenbach 2023

if nargin < 2 || isempty(k),      k      = 2; end
if nargin < 3 || isempty(rounds), rounds = k; end
icol = zeros(1,k); irow = zeros(1,k); Q = zeros(size(A,1),k); nr = k/rounds;
[opts.nr, opts.tol, opts.mindim, opts.maxdim, opts.v1] = deal(nr, 1e-5, 10, 30, ones(size(A,2),1)/sqrt(size(A,2)));

for i = 1:rounds
  ell = (i-1)*nr;
  if i == 1
    [~, V, U] = krylov_schur_svd(@(x,t) Bmv(x,t,A,[]), opts);
  else
    [~, V] = krylov_schur_svd(@(x,t) Bmv(x,t,A,Q(:,1:ell)), opts);
    V(icol(1:ell),:) = 0;
  end
  
  icol(ell+1:i*nr) = deim(V, nr);
  Q1 = A(:,icol(ell+1:i*nr));
  if i > 1, Q1 = Q1 - Q(:,1:ell)*(Q(:,1:ell)'*Q1); Q1 = Q1 - Q(:,1:ell)*(Q(:,1:ell)'*Q1); end
  [Q(:,ell+1:i*nr), ~] = qr(Q1, 0);
end

Q = zeros(size(A,2),k); opts.v1 = ones(size(A,1),1)/sqrt(size(A,1));
for i = 1:rounds
  ell = (i-1)*nr;
  if i > 1
    [~, U, ~] = krylov_schur_svd(@(x,t) Bmv(x,t,A',Q(:,1:ell)), opts);
    U(irow(1:ell),:) = 0;
  end
  irow(ell+1:i*nr) = deim(U, nr);
  Q1 = A(irow(ell+1:i*nr),:)';
  if i > 1, Q1 = Q1 - Q(:,1:ell)*(Q(:,1:ell)'*Q1); Q1 = Q1 - Q(:,1:ell)*(Q(:,1:ell)'*Q1); end
  [Q(:,ell+1:i*nr), ~] = qr(Q1, 0);
end
M = A(:,icol) \ (A / A(irow,:));

function y = Bmv(x, t, A, Q)
if t == 0
  y = A*x; if ~isempty(Q), y = y - Q*(Q'*y); end
else
  if isempty(Q), y = A'*x; else, y = x - Q*(Q'*x); y = A'*y; end
end
