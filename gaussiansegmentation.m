
clear all
%close all old figures each time script is executed
close all
%create struct with file info for training images
names=dir('./test_images/*.jpg');
%iterate through training images
for i = 1:length(names)
  clearvars -except names i
  
   figure(i)

  %relative path must include directory name 
  filename = strcat('./test_images/',names(i).name);
  %read current image
  pic = imread(filename); 
  imshow(pic)

  % use my functions to calculate covariance and average
  avg = findavg(pic);
  sigma = covariance(pic);
  %show current image
  
  %find gaussian posterior given sigma and mu
  %first set prior probability for the color orange
  porange=0.5;
  threshold(pic,porange,sigma,avg)
  posterior(pic(1,1,:),sigma,avg)
 pause(1) 
end
function threshold(img,prior,sig,mu)
sz=size(img);
orangeptsx=[];
orangeptsy=[];

    for i = 1:sz(1)
        for j = 1:sz(2)
           post = posterior(img(i,j,:),sig,mu);
           if post*prior<0.0000000000000000000005
            orangeptsx=horzcat(orangeptsx,j);
            orangeptsy=horzcat(orangeptsy,i);
           end
        end
    end
numpts=length(orangeptsx)
hold on
plot(orangeptsx,orangeptsy,'.')

end
function post = posterior(pixel,s,m)
    px=reshape(double(pixel),3,1);
    amp=1/(sqrt((2*pi)^3*det(s)));
    fun=exp((-0.5)*transpose(px-m)*inv(s)*(px-m));
    post=amp*fun;
end

function covar = covariance(orig)
  %find covariance in R^3x1
   img=double(orig);
   sz=size(img);
   avg = findavg(img);
   N=sz(1)*sz(2);
   s=zeros(3);
   for i =1:sz(1)
       for j = 1:sz(2)
           px=reshape(img(i,j,:),3,1);
          s=s+((px-avg)*(transpose(px-avg)));
       end
   end
   covar=s./N;
end
function avg = findavg(img)
sz=size(img);
  %find average in R^3x1
   N=sz(1)*sz(2);
   mu = [0; 0; 0];
   for j = 1:sz(1)
       for k = 1:sz(2)
           mu(1) = mu(1) + double(img(j,k,1));
           mu(2) = mu(2) + double(img(j,k,2));
           mu(3) = mu(3) + double(img(j,k,3));
       end
   end
  mu = mu/N;
  avg = mu;
end
function gaussianclustering(img)
end