function [A,A_labels] = gallery_curexps(matnr, m, n)

%GALLERY_CUR  CUR test examples
% function A = gallery_cur(matnr, m, n, p)
%
% Revision date: MAy 16, 2022
% (C) Perfect Gidisu, Michiel Hochstenbach 2020




if nargin < 1 || isempty(matnr), matnr = 1; end

A = [];
A_labels= [];
switch matnr
case 1  % Embree, Sorensen example 1
  if nargin < 2 || isempty(m), m = 100000; end
  if nargin < 3 || isempty(n), n = 300; end
  A = sparse([]);
  for j = 1:n
    x = sprand(m,1,0.025);  y = sprand(n,1,0.025);
    if j == 1, A = (x*(2/j))*y.';
    elseif j < 11, A = A + (x*(2/j))*y.'; else A = A + (x/j)*y.'; end
  end

case 2  % Embree, Sorensen example 2: TechTC term document data
    text=load('text');
    class=load('class');
    f=readtable('features.csv','ReadVariableNames',0);
    f=table2array(f);
    L=strlength(f);
    k = find(L>4);
    f_1=f(k);
    class(class<0)=0;
    text_f=text(:,k);
    A=normr(text_f);
    A_labels=f(k);
    
case 3 % Reuters 
  load('Reuters21578.mat');
  A=fea;

case 4 % g7jac100
load('g7jac100.mat')
A=Problem.A;

case 5 % invextr1_new
load('invextr1_new.mat')
A=Problem.A;


  

  
  
  

