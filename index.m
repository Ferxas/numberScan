load('Datos.mat');
 
figure 
plot(input,target);%Se traza una grafica utilizando los datos de entrada en el eje x y los datos objetivo de salida en el eje y
 
%no se preprocesa ya que el coseno osila entre 1 y -1
net=newff(input,target,10,{'tansig','purelin'},'trainlm'); %algoritmo de aprendisaje
 
%tansing permite de -1 a 1 mediante la funcion tangente hiperbolica
 
%purelin f(x)= x 
 
%muy pocas hacen underfiten
%
%HOLD OUT (Dividir el dataset en grupos de entrenamiento, prueba y validación
vectorSize=size(input,2);%en vecto sin 1x200 se obtiene el tamano del vector input numero de columna
trainRatio=0.8;    % 80% para entrenamiento
valRatio=0.1;      % 10% para validacion
testRatio=0.1;     % 10% para testeo
[trainInd,valInd,testInd] = dividerand(vectorSize,trainRatio,valRatio,testRatio);
 
%Indices del vector original
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.testInd  = testInd;
net.divideParam.valInd   = valInd;
 
% Modificación de parámetros de entrenamiento
% ========================================
%PARÁMETROS DE ENTRENAMIENTO
 
net.trainParam.show = 1;
net.trainParam.lr = 0.1;
net.trainParam.epochs = 300;  
net.trainParam.goal = 1e-9;    %Error minimo aceptable                          
                               %Controlar bien el error
%Ejecion de la red sin entrenar
outSE=sim(net,input);
figure
plot(input,outSE);
 
%entramiento de la red
[net]= train(net,input,target);
%ejecucion de la red entrenada
outCE= sim(net,input);
 
figure
subplot(3,1,1)
plot(input,outSE);
subplot(3,1,2);
plot(input,outCE);
subplot(3,1,3);
plot(input,target)
