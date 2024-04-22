function [irow, icol,M,rounds_col] = DADP_CUR_large(A, k, delta,l)

%CUR_DEIM_INOUT_DELTA_LARGE incurred CUR decomposition with SVD every iteration
% with two-sided projected residual; selects indices per round based on
% the decay pattern of the singular values
% for large-scale matrices
% function [irow, icol, M] = DADP_CUR_large(A, k, delta,l)
% k = desired number of indices 
% delta = upper bound on the relative magnitudes of the singular values
% l = upper bound on the number of singular vectors to be computed per
% iteration
% method =  limited svd
% C = A(:,icol);  R = A(irow,:)
%
% See also CUR_DEIM
%
% Revision date: June 29, 2023
% (C) Perfect Gidisu, Michiel Hochstenbach 2023

if nargin < 2 || isempty(k),     k     =   2; end
if nargin < 3 || isempty(delta), delta = 0.8; end
if nargin < 4 || isempty(l), l = 2; end
irow = []; icol = []; Qr = zeros(size(A,2),k);  Qc = zeros(size(A,1),k); rounds_col=[];
[opts.nr, opts.tol, opts.mindim, opts.maxdim, opts.v1] = deal(l, 1e-5 / sqrt(l), l, 2*l, ones(size(A,2),1)/sqrt(size(A,2)));

while length(irow) < k 
  if isempty(irow)
    [s, V, U] = krylov_schur_svd_relsigma(@(x,t) Bmv(x,t,A,[],[]), opts, delta);
  else
    [s, V, U] = krylov_schur_svd_relsigma(@(x,t) Bmv(x,t,A,Qc(:,1:length(icol)),Qr(:,1:length(irow))), opts, delta);
    U(irow,:) = 0; V(icol,:) = 0;
  end
  
  rounds_col=[rounds_col length(s)];
  tt = min([length(find(s >= delta*s(1))), k-length(irow), l]);
  ir = cur_deim(U(:,1:tt),tt);
  ic = cur_deim(V(:,1:tt),tt);
  irow = [irow ir];
  icol = [icol ic];
  ell = length(irow) - length(ir);
  Qc1 = A(:,icol(ell+1:ell+tt));
  Qr1 = A(irow(ell+1:ell+tt),:)';
  if ell > 0
    Qc1 = Qc1 - Qc(:,1:ell)*(Qc(:,1:ell)'*Qc1); Qc1 = Qc1 - Qc(:,1:ell)*(Qc(:,1:ell)'*Qc1);
    Qr1 = Qr1 - Qr(:,1:ell)*(Qr(:,1:ell)'*Qr1); Qr1 = Qr1 - Qr(:,1:ell)*(Qr(:,1:ell)'*Qr1);
  end
  [Qc(:,ell+1:ell+tt), ~] = qr(Qc1, 0);
  [Qr(:,ell+1:ell+tt), ~] = qr(Qr1, 0);  
end
M = A(:,icol) \ (A / A(irow,:));

function y = Bmv(x, t, A, Qc,Qr)
if t == 0
  y = A*x; if ~isempty(Qc), y = y -  Qc*(Qc'*(A*(Qr*(Qr'*x)))); end
else
  y = A'*x; if ~isempty(Qc), y = y - Qr*(Qr'*(A'*(Qc*(Qc'*x)))); end
end
