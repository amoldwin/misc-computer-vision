mat1 = (matfile('data/data1.mat'));
mat1 = mat1.('pts');

%load data from files
mat2 = (matfile('data/data2.mat'));
mat2 = mat2.('pts');

mat3 = (matfile('data/data3.mat'));
mat3 = mat3.('pts');
%close any open figures when re-executing
close all

%call my function to find the covariance matrices for each dataset
covariance(mat1);
covariance(mat2);
covariance(mat3);

% I know MATLAB has a function for this but this is an exercise in doing
% things on my own
function cov = covariance(data)
    %covariance in sample gets divided by n-1
    %treating our points as a sample as defined in tutorial
    Nminus1=length(data)-1;
    %average assuming the sample bias again
    avgx = sum(data(1,:))/Nminus1;
    avgy = sum(data(2,:))/Nminus1;
    %row vector to store the variances and covariances for now
    [xx, xy, yx, yy] = deal(0,0,0,0);
    %calculate sum
    for i = 1:length(data)
        x = data(1,i);
        y = data(2,i);
        xx = xx + (x-avgx)*(x-avgx);
        xy = xy + (x-avgx)*(y-avgy);
        yx = xy;
        yy = yy + (y-avgy)*(y-avgy);
    end
    figure;
    %plot data
    plot(data(1,:),data(2,:),'*', 'linewidth',5);
    %divide covariance matrix values by n-1 
    [xx,xy,yx,yy] = deal((xx/Nminus1), (xy/Nminus1),( yx/Nminus1), (yy/Nminus1));
    %return 2x2 matrix
    cov = [xx,xy;yx,yy];
    hold on
    %find eigenvectors, values
    [V,D] = eig(cov);
    %find length to normalize eigenvectors
    LV1 = norm([V(1,1) V(2,1)]);
    LV2 = norm([V(1,2) V(2,2)]);
    %twice the sqrt of the eigenvalues
    l1 = 2*sqrt(D(1,1));
    l2 = 2*sqrt(D(2,2));
    %eigenvectors scaled by twice the sqrt of corresponding eigenvalues
    v1 =[(V(1,1)/LV1)*l1,(V(2,1)/LV1)*l1];
    v2 =[(V(1,2)/LV2)*l2,(V(2,2)/LV2)*l2];
    %plot scaled vectors, centered at average x, y
    quiver(avgx-v1(1)/2,avgy-v1(2)/2,v1(1),v1(2),1,'linewidth',5 )
    quiver(avgx-v2(1)/2,avgy-v2(2)/2,v2(1),v2(2),1,'linewidth',5 )
    %need axes set to same scale to see that vectors are orthogonal
    axis square
    axis equal
    %create latex string for textbox with covariance matrix
    str = sprintf('$$\\Sigma =  \\pmatrix{%d & %d \\cr %d & %d} $$',xx,xy,yx,yy);
    annotation('textbox',[.2 .5 .3 .3],'Interpreter','latex','String',str,'FitBoxToText','on');
    %show item labels
    legend('data','2*sqrt(\lambda_1)v_1 centered at mean','2*sqrt(\lambda_1)v_2 centered at mean')
    xlabel('x')
    ylabel('y')
end

