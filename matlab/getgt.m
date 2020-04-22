close all
clear all

imageFileBasePath = '/media/usr134/本地磁盘/syj/0218pic/test/1/input/';%file path to read the images of mutiple objects 
fileForm ='*.jpg';%format of the images

trainPath = '/media/usr134/本地磁盘/syj/0227face/test/1/gt/';
testPath='/media/usr134/本地磁盘/syj/0227face/test/1/input/';
imageFilePath = strcat(imageFileBasePath);%,TestorTrain,'\');
files = dir(fullfile(imageFilePath,fileForm)); 
len1 = size(files,1);
for numSimulation = 1:126
    filename = strcat(imageFilePath,files(numSimulation).name);

    b=filename(end-5:end-4);
    c=b;
    b=str2num(b);

    splitedStr = strsplit(filename,'/');
    outfile = cell2mat(splitedStr(end));

    initialImage = imread(filename);
%     initialImage=rgb2gray(initialImage);
    [sizeX,sizeY,sizeZ]=size(initialImage);
%     if (sizeX~=200)
%         continue
%     end
%     zeropic=zeros(sizeX,sizeX);
%     zeropic=im2uint8(zeropic);
%     zeropic(1:sizeX,11:10+180)=initialImage;
%     zeropic=imresize(zeropic,[256,256]);
    if(sizeX==256)
        if((b>15)&&(b<18))
            finalPath = strcat(testPath,outfile);
            imwrite(initialImage,finalPath);
%         else
%             finalPath = strcat(trainPath,outfile);
%             imwrite(zeropic,finalPath); 
        end
    end
    
%     initialImage=imresize(initialImage,[256,256]);
%     output=initialImage((1920-256)/2:(1920+256)/2-1,(1200-256)/2:(1200+256)/2-1);
%     output=initialImage(912:912+255,472:472+255);
%     output=mat2gray(output);
%     outfile = sprintf('%d_%d.bmp', qz,b);
%     imwrite(output,[cameraImagePath,num2str( a) '_'num2str(b) '_' num2str(c) '.bmp'],'bmp');
%     imwrite(output,[cameraImagePath, num2str(a) '_' num2str(b) '_' num2str(c) '.bmp'],'bmp');
    
end