function icol = deim(V,k)

% Reference: Chaturantabut and Sorensen 2010, Embree and Sorensen, 2016
% V is singular vectors
% k desired number of indices

icol=zeros(1,k);
[~, icol(1)] = max(abs(V(:,1)));
for j = 2:k
   V(:,j) = V(:,j) - V(:,1:j-1) * (V(icol(1:j-1),1:j-1) \ V(icol(1:j-1),j));
   [~, icol(j)] = max(abs(V(:,j)));
end
