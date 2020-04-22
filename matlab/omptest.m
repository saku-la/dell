clc;clear

%  读文件
X=imread('/media/usr134/本地磁盘/syj/0227face/more/train/1/input//9326871.1.jpg');
X=double(X);
[a,b]=size(X);

%  小波变换矩阵生成
% ww=DWT(a);

%  小波变换让图像稀疏化（注意该步骤会耗费时间，但是会增大稀疏度）
% X1=ww*sparse(X)*ww';
X1=X;
% X1=full(X1);

%  随机矩阵生成
M=190;
R=rand(256,256);
    for ii=1:256
        for jj=1:256
            R(ii,jj)=R(ii,jj)*100;
            if (R(ii,jj)>33)
                R(ii,jj)=1;
            else
                R(ii,jj)=0;
            end
        end
    end
% R=mapminmax(R,0,255);
% R=round(R);

%  测量值
Y=R*X1;
B=[ ];
%  OMP算法
%  恢复矩阵
X2=zeros(a,b); 
%  按列循环
for i=1:b 
    %  通过OMP，返回每一列信号对应的恢复值（小波域）
    [rec,A]=omp(Y(:,i),R,a);
    %  恢复值矩阵，用于反变换
    X2(:,i)=rec;
    B=[B A];
end


%  原始图像
% figure(1);
% imshow(uint8(X));
% title('原始图像');
% 
% %  变换图像
% figure(2);
% imshow(uint8(X1));
% title('小波变换后的图像');
% 
% %  压缩传感恢复的图像
% figure(3);
% %  小波反变换
% % X3=ww'*sparse(X2)*ww; 
% X3=X2;
% X3=full(X3);
% imshow(uint8(X3));
% title('恢复的图像');
% 
% %  误差(PSNR)
% %  MSE误差
% errorx=sum(sum(abs(X3-X).^2));        
% %  PSNR
% psnr=10*log10(255*255/(errorx/a/b))   


%  OMP的函数
%  s-测量；T-观测矩阵；N-向量大小
function [hat_y,A]=omp(s,T,N)
Size=size(T);                                     %  观测矩阵大小
M=Size(1);                                        %  测量
hat_y=zeros(1,N);                                 %  待重构的谱域(变换域)向量                     
Aug_t=[];                                         %  增量矩阵(初始值为空矩阵)
r_n=s;                                            %  残差值

for times=1:M;                                  %  迭代次数(稀疏度是测量的1/4)
    for col=1:N;                                  %  恢复矩阵的所有列向量
        product(col)=abs(T(:,col)'*r_n);          %  恢复矩阵的列向量和残差的投影系数(内积值) 
    end
    [val,pos]=max(product);                       %  最大投影系数对应的位置
    Aug_t=[Aug_t,T(:,pos)];                       %  矩阵扩充
    T(:,pos)=zeros(M,1);                          %  选中的列置零（实质上应该去掉，为了简单我把它置零）
    aug_y=(Aug_t'*Aug_t)^(-1)*Aug_t'*s;           %  最小二乘,使残差最小
    r_n=s-Aug_t*aug_y;                            %  残差
    pos_array(times)=pos;                         %  纪录最大投影系数的位置
    
    if (norm(r_n)<0.9)                            %  残差足够小
        break;
    end
end
hat_y(pos_array)=aug_y;  
A=Aug_t;
end