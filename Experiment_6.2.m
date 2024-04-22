% Experiment #6.2.
% case 1 synthetic
% case 2 TechTC term document data
% case 3 Reuters
% case 4 g7jac100
% case 5 invextr1_new
[A, Feat] = gallery_curexps(4);

tol=1e-2;

delta=0.8;
cons= 10;
e=normest(A);
c=20:10:60;
[U,~,V]=svds(A,60);
for i=1:length(c)
    k= c(i); 
    l=k/cons;
    
    irow0 = deim(U(:,1:k),k);
    icol0 = deim(V(:,1:k),k); 
    M0=A(:,icol0)\A/A(irow0,:);
    err0(i)=normest(A-(A(:,icol0)*M0*A(irow0,:)))/e;
    clear   M0

    [irow1, icol1,M1] = CADP_CUR_large(A, k,cons);
    err1(i)=normest(A-(A(:,icol1)*M1*A(irow1,:)))/e;
    clear  M1
    
    [irow2, icol2,M2] = DADP_CUR_large(A, k,delta,l); 
    err2(i)=normest(A-(A(:,icol2)*M2*A(irow2,:)))/e;
    clear  M2 
    
    [irow3, icol3,M3] = CADP_CX_large(A, k,cons);
    err3(i)=normest(A-(A(:,icol3)*M3*A(irow3,:)))/e;
    clear  M3 
   
    [irow4, icol4,M4] = DADP_CX_large(A, k,delta,l);
    err4(i)=normest(A-(A(:,icol4)*M4*A(irow4,:)))/e;
    clear M4

    irow5 = maxvol(U(:,1:k),tol);
    icol5 = maxvol(V(:,1:k),tol);
    M5=A(:,icol5)\A/A(irow5,:);
    err5(i)=normest(A-(A(:,icol5)*M5*A(irow5,:)))/e;
    clear  M5 

    irow6 = qdeim(U(:,1:k),k);
    icol6 = qdeim(V(:,1:k),k);
    M6=A(:,icol6)\A/A(irow6,:);
    err6(i)=normest(A-(A(:,icol6)*M6*A(irow6,:)))/e;
    clear  M6 

    
end





plot(l,err0,'-d')
hold on;
plot(l,err1,'-*')
plot(l,err2,'-o')
plot(l,err3,'-s')
plot(l,err4,'-v')
plot(l,err5,'->')
plot(l,err6,'-^')
ylabel('|| A - C U R ||/ || A||','fontweight','bold','fontsize',16)
xlabel('k','fontweight','bold','fontsize',16);
  
legend('DEIM-SEQ','CADP-CUR','DADP-CUR','CADP-CX','DADP-CX','MaxVol','QDEIM')
