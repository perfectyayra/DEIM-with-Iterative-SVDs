function [irow, icol,M] = cur_deim_inout_delta_small(A, k, delta,l)

%CUR_DEIM_INOUT_DELTA_small incurred CUR decomposition with SVD every iteration
% with two-sided projected residual; selects indices per round based on
% the decay pattern of the singular values
% for small-scale matrices
% function [irow, icol, M] = cur_deim_inout_delta_small(A, k, delta,l)
% k = desired number of indices 
% delta = upper bound on the relative magnitudes of the singular values
% l = upper bound on the number of singular vectors to be computed per
% iteration
% method = either full svd or limited svd
% C = A(:,icol);  R = A(irow,:)
%
% See also CUR_DEIM
%
% Revision date: June 29, 2023
% (C) Perfect Gidisu, Michiel Hochstenbach 2023

if nargin < 2 || isempty(k),     k     =   2; end
if nargin < 3 || isempty(delta), delta = 0.8; end
if nargin < 4 || isempty(l), l = 2; end
irow = []; icol = []; M = [];

while length(irow) < k
  if isempty(irow)
    [U,S,V] = svd(A,0); s = diag(S);
  else
    B = A - A(:,icol) * M  * A(irow,:);
    [U,S,V] = svd(B,0); s = diag(S);
    U(irow,:) = 0;  V(icol,:) = 0;
  end
  tt = min([length(find(s >= delta*s(1))), k-length(irow),  l]);
  ir= cur_deim(U(:,1:tt),tt);
  ic= cur_deim(V(:,1:tt),tt);
  irow = [irow ir];
  icol = [icol ic];
  M = A(:,icol) \ (A / A(irow,:)); 
end
