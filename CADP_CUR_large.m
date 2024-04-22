function [irow, icol,M] = CADP_CUR_large(A, k,rounds)

%CUR_DEIM_INOUT_LARGE incurred CUR decomposition with SVD every iteration with
%two-sided projected residual; selects constant number of indices per round
% function [irow, icol, M] = CADP_CUR_large(A, k, rounds)
% for large-scale matrices
% k = desired number of indices 
% rounds = number of iterations ;  rounds | k
% method = limited svd
% C = A(:,icol);  R = A(irow,:)
%
% See also CUR_DEIM
%
% Revision date: June 29, 2023
% (C) Perfect Gidisu, Michiel Hochstenbach 2023

if nargin < 2 || isempty(k),      k      = 2; end
if nargin < 3 || isempty(rounds), rounds = k; end
irow = zeros(1,k); icol = irow;   nr = k/rounds;
Qr = zeros(size(A,2),k);  Qc = zeros(size(A,1),k);
[opts.nr, opts.tol, opts.mindim, opts.maxdim, opts.v1] = deal(nr, 1e-5, 10, 30, ones(size(A,2),1)/sqrt(size(A,2)));
for i = 1:rounds
  ell = (i-1)*nr;
  if i == 1
    [~, V, U] = krylov_schur_svd(@(x,t) Bmv(x,t,A,[],[]), opts);
  else
    [~, V, U] = krylov_schur_svd(@(x,t) Bmv(x,t,A,Qc(:,1:ell),Qr(:,1:ell)), opts);
    U(irow(1:ell),:) = 0; V(icol(1:ell),:) = 0; 
  end
  irow(ell+1:i*nr) = cur_deim(U,nr);
  icol(ell+1:i*nr) = cur_deim(V,nr);
  Qc1 = A(:,icol(ell+1:i*nr));
  Qr1 = A(irow(ell+1:i*nr),:)';
  if i > 1
    Qc1 = Qc1 - Qc(:,1:ell)*(Qc(:,1:ell)'*Qc1); Qc1 = Qc1 - Qc(:,1:ell)*(Qc(:,1:ell)'*Qc1);
    Qr1 = Qr1 - Qr(:,1:ell)*(Qr(:,1:ell)'*Qr1); Qr1 = Qr1 - Qr(:,1:ell)*(Qr(:,1:ell)'*Qr1);
  end
  [Qc(:,ell+1:i*nr), ~] = qr(Qc1, 0);
  [Qr(:,ell+1:i*nr), ~] = qr(Qr1, 0);  
end
M = A(:,icol) \ (A / A(irow,:));

function y = Bmv(x, t, A, Qc, Qr)
if t == 0
  y = A*x; if ~isempty(Qc), y = y -  Qc*(Qc'*(A*(Qr*(Qr'*x)))); end
else
  y = A'*x; if ~isempty(Qc), y = y - Qr*(Qr'*(A'*(Qc*(Qc'*x)))); end
end

