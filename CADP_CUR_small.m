function [irow, icol,M] = CADP_CUR_small(A, k,rounds)

%CUR_DEIM_INOUT_SMALL incurred CUR decomposition with SVD every iteration with
%two-sided projected residual; selects constant number of indices per round
% for small-scale matrices
% function [irow, icol, M] = CADP_CUR_small(A, k, rounds)
% k = desired number of indices 
% rounds = number of iterations;  rounds | k
% method = either full svd or limited svd
% C = A(:,icol);  R = A(irow,:)
%
% See also CUR_DEIM
%
% Revision date: June 28, 2023
% (C) Perfect Gidisu, Michiel Hochstenbach 2023

if nargin < 2 || isempty(k),      k      = 2; end
if nargin < 3 || isempty(rounds), rounds = k; end
irow = zeros(1,k); icol = irow;  M = [];  nr = k/rounds;

for i=1:rounds
  ell = (i-1)*nr;
  if i == 1
    [U, ~, V] = svd(A,0); U = U(:,1:nr); V = V(:,1:nr);
  else
    B = A - A(:,icol(1:ell))*M*A(irow(1:ell),:); 
    [U, ~, V] = svd(B,0);  U = U(:,1:nr);  V = V(:,1:nr);
    U(irow(1:ell),:) = 0; V(icol(1:ell),:) = 0; 
  end
  irow(ell+1:i*nr) = deim(U,nr);
  icol(ell+1:i*nr) = deim(V,nr);
  M = A(:,icol(1:i*nr)) \ (A / A(irow(1:i*nr),:));
end


