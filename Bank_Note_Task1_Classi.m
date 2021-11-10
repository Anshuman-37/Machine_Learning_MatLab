%%% TASK 1 
% Importing Data
X = BankNoteAuthentication_x; %Features for Extraction
Y = BankNoteAuthentication_y ; %Target Vector

%Visualising the Data
%Refrence https://www.mathworks.com/help/matlab/titles-and-labels.html
figure,boxplot(X,'Notch','on','Labels',{'Variance','Skewness','Curtosis','Entropy'});

%Task 1 Classification Train SVM with Linear Kernel
 
%Refrences https://moodle.nottingham.ac.uk/pluginfile.php/8161069/
%mod_resource/content/1/COMP3009MLE_AssessedLab1_SVMs_2021.pdf 

%Values for the params 
Name = 'Linear';
Value = 'Boxconstraint';

%Training the model
MDLIN = fitcsvm(X,Y,'KernelFunction',Name,Value,1);

%Plotting the Confussion matrix for the model and the accurracy
%Reference https://uk.mathworks.com/help/stats/confusionmat.html 
Ypred = predict(MDLIN,X);
C = confusionmat(Y,Ypred);
confusionchart(C)

%Accruacy of the model AUC ans ROC curve 
%Refrence https://stackoverflow.com/questions/2553505120
accuracy = sum(Y == Ypred) / numel(Y);
accuracyPercentage = 100*accuracy;
fprintf('The accuracy of the model is %f \n', accuracyPercentage);