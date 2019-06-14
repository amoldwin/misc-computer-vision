%load the 3 data files. X and Y are stored as the two data rows
mat1 = (matfile('data/data1.mat'));
mat1 = mat1.('pts');

mat2 = (matfile('data/data2.mat'));
mat2 = mat2.('pts');

mat3 = (matfile('data/data3.mat'));
mat3 = mat3.('pts');
%close old plots when running again
close all
%-OLS function calls RANSAC and Tikhonov functions automatically at end
%-the second arg here is the tuning parameter for the Tikhenov
%regularization
leastsquares(mat1,0.0000003)
leastsquares(mat2,0.0000007)
leastsquares(mat3,0.000005)


%function for plotting line based on tikhonov regularization 
function tikh = tikhonov(data,tune)
%separate into 2 vectors for convenience
xvals = data(1,:);
yvals = data(2,:);

%-Set up matrices in closed form solution formulation for Tikhonov
%regularization
X = [transpose(data(1,:)) ones([length(data),1])];
Y = transpose(yvals); 


%commented out code to see effect of lambda on condition number

%minterm = @(L)cond((transpose(X)*X)+L*eye(2));
%condx=[-50:49]
%condy=[1:100]
%figure
%for i = (1:100)
%condy(i)=minterm(condx(i))
%end
%plot(condx,condy)
%minterm(100)
%options = optimset('MaxFunEvals',1000);
%f=fminsearch(minterm,10)

%anonymous function to extract the fit from closed form solution
tik = @(X,Y)(inv(transpose(X)*X)+tune*eye(2))*(transpose(X)*Y);
B = tik(X,Y);
[m,b]=deal(B(1),B(2));

%anonymous function for the ditted line
lineeq= @(x)x.*(m)+(b);
%plot the line equation from the min x value in the data to the max
[x1,x2] = deal(min(xvals),max(xvals)) ;
[y1,y2] =deal(lineeq(x1),lineeq(x2));
plot([x1; x2],[y1; y2],'linewidth',5);
hold on 
end

function ran = ransac(data,ratio)
%define most parameters here, hard code very high P so this works for all
%datasets to avoid tuning
    s = 2;
    t = 10;
    p=0.99999;
    r=ratio;
    %calculate number of iterations based on previous params
    N = ceil(log(1-p)/log(1-(1-r)^s))
    xvals = data(1,:);
    yvals = data(2,:);
    %inliers are a subset of data initialized to the whole dataset and
    %recalculated after each fit
    xinliers = xvals;
    yinliers =  yvals;
    %here we keep track of the best slope and intercept achieved in the fit
    %with highest inlier ratio so far at each iteration
    bestmb=[0,0];
    bestd = 0;
    %iterate
    for i = 1:N
        %pick two points at random indices in the data
        p1=randi(length(xinliers));
        p2=randi(length(xinliers));
        %if they are the same, change the second point
        while(p1==p2)
            p2=randi(length(xinliers));
        end
        %find teh x and y values from the data at these two indices
        x1=xinliers(p1);
        y1=yinliers(p1);
        x2=xinliers(p2);
        y2=yinliers(p2);
        %the current fit is the line through these points
        m=(y2-y1)/(x2-x1);
        lineeq=@(x)m*(x-x1)+y1;
        
        %commented out code to plot each line to see fits get better
        %[minx,maxx] = deal(min(xvals),max(xvals)) ;
        %[miny,maxy] =deal(lineeq(minx),lineeq(maxx));
        %plot([minx; maxx],[miny; maxy],'linewidth',5)  
        %legend(gca,'show')
        %count the inliers
        
        %calculate number of datapoints that should be inliers
        d = (1-r)*length(xvals);
        %count number that are within error tolerance from current fit
        numinliers = 0;
        for j = 1:length(xvals)
            if abs(yvals(j)-lineeq(xvals(j)))<=t
                numinliers=numinliers+1;
            end
        end
        %see if this is better than all previous fits, if so update best
        if numinliers>bestd
            bestd=numinliers;
            bestmb(1)=m;
            bestmb(2)=y1-m*x1;
            
        end
        %filter out outliers and reform inliers array
        if numinliers/(length(xvals)) >= d
            newxinliers = zeros(1,numinliers);
            newyinliers = zeros(1,numinliers);
            filled=0;
            for j = 1:length(xvals)
                if yvals(j)-lineeq(xvals(j))<=t
                    filled=filled+1;
                    newxinliers(filled)=xvals(j);
                    newyinliers(filled)=yvals(j);
                end
            end
            %update inliers arrays outside loop scope
            xinliers=newxinliers;
            yinliers=newyinliers;
        end
    end
    %equation of best fit after iterating is done
    lineeq=@(x)bestmb(1)*x+bestmb(2);
    %plot best fit from minimum x in data to max x
        [minx,maxx] = deal(min(xvals),max(xvals)) ;
        [miny,maxy] =deal(lineeq(minx),lineeq(maxx));
        plot([minx; maxx],[miny; maxy],'linewidth',5)
        hold on
        %legend(strcat('N =', strsplit(num2str([1:N]))))
        %plot(xvals,yvals,'*','linewidth',5)

end

function lsq = leastsquares(data,tune)
%new plot each time this is called
figure
xvals = data(1,:);
yvals = data(2,:);

%-re-form the data into matrices of the form of the closed solution for OLS
%optimization problem
X = [transpose(data(1,:)) ones([length(data),1])];
Y = transpose(yvals); 

%calculate OLS solution matrix beta^ and extract slope, intercept 
ols = @(X,Y)(inv(transpose(X)*X))*(transpose(X)*Y);
B = ols(X,Y);
[m,b]=deal(B(1),B(2));
%[U,S,V]=svd(A);

%equation of fitted line
lineeq= @(x)x.*(m)+(b);
%x and y values to plot the line
[x1,x2] = deal(min(xvals),max(xvals)) ;
[y1,y2] =deal(lineeq(x1),lineeq(x2));
plot(xvals,yvals,'*','linewidth',5)
hold on
plot([x1; x2],[y1; y2],'linewidth',5);
hold on 
%call other functions to include them in current plot
tikhonov(data,tune)
ransac(data,0.4)
%label each plotted item
legend('data','Ordinary Least Squares Without Regularization', strcat('OLS with Tikhonov Regularization, \lambda = ',num2str(tune)),'RANSAC with t=0.4')
end

