% código para cargar los datos de entrenamiento...

% imageDir = 'dataset'; % directorio que contiene las imágenes
% imageFiles = dir(fullfile(imageDir, '*.jpg')); % obtener la lista de archivos
% numImages = numel(imageFiles); % tomará el número de imágenes del directorio
% input = cell(numImages, 1); % matriz de celdas para guardas imágenes

% for i = i:numImages
    % imagePath = fullfile(imageDir, imageFiles(i).name);
    % input{i} = imread(imagePath);
% end

% ------ 

% image1 = imread('dataset/0/4924.jpg'); % imagen 0
% image2 = imread('dataset/1/5742.jpg'); % imagen 1
% ...

input = [vectorizeImage(image1); vectorizeImage(image2)]; 
target = [target1; target2];

net = newff(input, target, 10, {'tansig', 'purelin'}, 'trainlm');
net.divideFcn = 'divideind';
[trainInd, valInd, testInd] = dividerand(size(input, 2), trainRatio, valRatio, testRatio);
net.divideParam.trainInd = trainInd;
net.divideParam.testInd = testInd;
net.divideParam.valInd = valInd;

net.trainParam.show = 1;
net.trainParam.lr = 0.1;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-9;

net = train(net, input, target);

output = sim(net, input);

figure;
subplot(3, 1, 1);
plot(input, output);
title('Mi red neuronal...');
subplot(3, 1, 2);
plot(input, target);
title('Objetivo');
subplot(3, 1, 3);
plot(input, output);
hold on;
plot(input, output);
legend('Salida', 'Objetivo');
title('Comparación entre salida y objetivo');