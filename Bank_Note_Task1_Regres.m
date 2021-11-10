%TASK 1 Regression


% Importing Data
X = concretedata_x; %Features for Extraction
Y = concretedata_y ; %Target Vector


%Visualising the Data
%Refrence https://www.mathworks.com/help/matlab/titles-and-labels.html
figure,boxplot(X,'Notch','on','Labels',{'Cement (component 1)',...
    'Blast Furnace Slag (component 2))','Fly Ash (component 3)'...
    ,'Water (component 4)','Superplasticizer (component 5)',...
    'Coarse Aggregate  (component 6)','Fine Aggregate (component 7)'...
    ,'Age (day)'});

 

%Task 1 Train SVM with Linear Kernel

 

%Refrences https://moodle.nottingham.ac.uk/pluginfile.php/8161069/
%mod_resource/content/1/COMP3009MLE_AssessedLab1_SVMs_2021.pdf 

%Training the model for Regression
Mdl = fitrsvm(X,Y,'Epsilon',0.0000005)

%Plotting the real values with respect to the predicted values 
%Refrence https://uk.mathworks.com/help/stats/fitrsvm.html
Ypred = predict(Mdl,X);
plot(Y,Y,'o',Y,Ypred,'x')
legend('Y','YPredictions')

%The Mean Squared Error for the Regression model
%Refrence https://uk.mathworks.com/help/stats/regressionsvm-class.html
%Refrence https://www.thoughtco.com/how-to-calculate-percent-error-609584
M_pre = mean(Ypred,'all');
M_true = mean(Y,'all');
Per_error = (M_pre - M_true)/(M_true)*100;
fprintf('The accuracy for the model is %f\n',(100-Per_error));

 