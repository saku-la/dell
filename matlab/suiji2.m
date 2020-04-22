%
%NJUST syj 2020.4.21
%
%
close all
clear all

numImage=1;
inputsize=60;%中心图片大小
imageFilePath = 'D:\sss\za\3500\';
                   
outImageFilePath = 'D:\sss\za\3.18\';
fileForm = '*.jpg';
 
files = dir(fullfile(imageFilePath,fileForm)); 
len1 = size(files,1);
indexRandomInput = ceil(len1 * rand(numImage,2));
t=clock;
for numSimulation = 1 : numImage
    tic;
    filename1 = strcat(imageFilePath,files(indexRandomInput(numSimulation,1)).name);  
    initialImage1 = imread(filename1);
    filename2 = strcat(imageFilePath,files(indexRandomInput(numSimulation,2)).name);
    initialImage2 = imread(filename2);
    inputImageTemp1 = initialImage1(:,:,1);
    inputImageT1 = double(imresize(inputImageTemp1,[inputsize,inputsize]));
    inputImageTemp2 = initialImage2(:,:,1);
    inputImageT2 = double(imresize(inputImageTemp2,[inputsize,inputsize]));
    a=rand(1);
    b=rand(1);
    a1=rand(1);
    b1=rand(1);
    c=1-(inputsize/256);
    d=inputsize-1;
    while a>c
        a=rand(1);
        if a<c
            break
        end
    end
    while b>c
        b=rand(1);
        if b<c
            break
        end
    end
    while a1>c
        a1=rand(1);
        if a1<c
            break
        end
    end
    while b1>c
        b1=rand(1);
        if b1<c
            break
        end
    end
    zeropic=zeros(256,256);
    zeropic(ceil(a*256):ceil(a*256+d),ceil(b*256):ceil(b*256)+d)=inputImageT1;
    roipic=zeropic(ceil(a1*256):ceil(a1*256+d),ceil(b1*256):ceil(b1*256)+d);
    zeropic(ceil(a1*256):ceil(a1*256+d),ceil(b1*256):ceil(b1*256)+d)=inputImageT2+roipic;
    zeropic(zeropic>255) = 250;
    
  
    zeropic = mat2gray(zeropic);
    
    outfile = sprintf('%d_%05d.bmp', inputsize, numSimulation);    
    
    outputImageName = strcat(outImageFilePath,outfile);
    imwrite(zeropic,outputImageName);
    
end
