function [irow, icol, M] = CADP_CX_small(A, k, rounds)

%CX_DEIM_INOUT_SMALL incurred CUR decomposition with SVD every iteration with
% one-sided projected residual; selects constant number of indices per round
% function [irow, icol, M] = CADP_CX_small(A, k, rounds)
% for small-scale matrices
% k = desired number of indices 
% rounds = number of iterations, need: rounds | k
% C = A(:,icol);  R = A(irow,:)
%
% See also CUR_DEIM
%
% Revision date: June 29, 2023
% (C) Perfect Gidisu, Michiel Hochstenbach 2023

if nargin < 2 || isempty(k),      k      = 2; end
if nargin < 3 || isempty(rounds), rounds = k; end  % 1 vector per round
icol = zeros(1,k); irow = zeros(1,k); M = []; nr = k / rounds;

for i = 1:rounds
  ell = (i-1)*nr;
  if i == 1
    [U, ~, V] = svd(A,0); V = V(:,1:nr); U = U(:,1:nr);
  else
    B = A - A(:,icol(1:ell)) * M;
    [~, ~, V] = svd(B,0); V = V(:,1:nr);
    V(icol(1:ell),:) = 0;
  end
  icol(ell+1:i*nr) = deim(V, nr);
  M = A(:,icol(1:i*nr)) \ A;
end

for i = 1:rounds
  ell = (i-1)*nr;
  if i > 1
    B = A - M * A(irow(1:ell),:);
    [U, ~, ~] = svd(B,0); U = U(:,1:nr);
    U(irow(1:ell),:) = 0; 
  end
  irow(ell+1:i*nr) = deim(U, nr);
  M = A / A(irow(1:i*nr),:);
end
M = A(:,icol) \ M;
