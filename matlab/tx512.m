close all
clear all

imageFileBasePath = '/home/usr134/GEL_DATA/20200515FaceData/gt/';%file path to read the images of mutiple objects 
fileForm ='*.bmp';%format of the images

trainPath = '/media/usr134/本地磁盘/syj/0227face/realface/80/train/1/gt/';
testPath='/media/usr134/本地磁盘/syj/0227face/realface/80/test/1/gt/';
imageFilePath = strcat(imageFileBasePath);%,TestorTrain,'\');
files = dir(fullfile(imageFilePath,fileForm)); 
len1 = size(files,1);
numpic=20;
len2=numpic*15;
for numSimulation = 1:500
    filename = strcat(imageFilePath,files(numSimulation).name);

    

    splitedStr = strsplit(filename,'/');
    outfile = cell2mat(splitedStr(end));
    initialImage = imread(filename);
    outnum=outfile(1:6);
    outnum=str2num(outnum);
    num=mod(outnum,20);
    if num==0
        num=20;
    end
    if num<19
%         finalPath = strcat(trainPath,outfile);
%         imwrite(initialImage,finalPath);
        imwrite(initialImage,[trainPath,num2str(outnum,'%06d') '.bmp'],'bmp');
    else
%         finalPath = strcat(testPath,outfile);
%         imwrite(initialImage,finalPath);
        imwrite(initialImage,[testPath,num2str(outnum,'%06d') '.bmp'],'bmp');
    end
   


end