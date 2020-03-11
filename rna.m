clc
#pkg load image
% Load CSV files
A = csvread('Treino/DataA.csv');
A(1)=1;
#A=A(2:end);
B = csvread('Treino/DataB.csv');
B(1)=1;
w=zeros(1,1025);
x=[A;B];

t=[-1 1;1 1]
linhas =2

#treino
for i=1:length(linhas)
  xi=x(i,:);
  ti=t(i,:);
  ti=transpose(ti);
  dw=ti*xi;
  w=w+dw;
endfor



#teste
for i=1:length(linhas)
  xi=x(i,:);
  xit=transpose(xi);
  net=w*xit
    


endfor


#image=imread('Treino/A.png');
#image=im2bw(image);
#reshape(image,1,[])

