close all
clear all

imageFileBasePath = '/home/usr134/GEL_DATA/20200514doublechar/train/1/input/';%file path to read the images of mutiple objects 
fileForm ='*.bmp';%format of the images

trainPath = '/media/usr134/本地磁盘/syj/0218pic/real2char/train/1/input/';
% testPath='/media/usr134/本地磁盘/syj/0227face/realface/80/test/1/gt/';
imageFilePath = strcat(imageFileBasePath);%,TestorTrain,'\');
files = dir(fullfile(imageFilePath,fileForm)); 
len1 = size(files,1);
numpic=20;
len2=numpic*15;
for numSimulation =1:7000
    filename = strcat(imageFilePath,files(numSimulation).name);

    

    splitedStr = strsplit(filename,'/');
    outfile = cell2mat(splitedStr(end));
    initialImage = imread(filename);
    initialImage=mat2gray(initialImage);
    outnum=outfile(1:4);
    outnum=str2num(outnum);
    imwrite(initialImage,[trainPath,num2str(outnum,'%06d') '.bmp'],'bmp');
   


end