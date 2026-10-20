function SVMresults = classifySVM_18(trainTable, testTable, varargin)

% SVM classification with k-fold cross-validation on training data,
% then evaluation on test data.
%
% INPUTS:
%   trainTable : feature table with FileName, Label, feature columns
%   testTable  : feature table with FileName, Label, feature columns
%
% OPTIONAL NAME-VALUE INPUTS:
%   'KFold'             : number of CV folds                (default: 10)
%   'KernelFunction'    : 'rbf', 'linear', 'polynomial'     (default: 'rbf')
%   'BoxConstraint'     : C [0.1,1,10,100]                  (default: 2)
%   'KernelScale'       : Gamma (in 'rbf') [0.1,1,10, 50]   (default: 11)
%   'PolynomialOrder'   : degree (in 'polynomial') [2,3,4]  (default: 3)
%   'Standardize'       : true/false                        (default: false)
%
% OUTPUT:
%   results : struct with CV (per-fold + summary) and test metrics
%
% EXAMPLE:
%   SVMresults = classifySVM_15(trainData, testData)
%   results = classifySVM(trainTable, testTable);
%   results = classifySVM(trainTable, testTable, 'KFold', 10, 'KernelFunction', 'linear');

    %% ---- Tuning Parameters ----
    % Under-sampling Factor for unbalanced dataset
    underSamplingFactor = 2;    % value of 1.5 means 60%-40%
    classFactor = 5;              % affects normal class

    %% ---- Parse inputs ----
    p = inputParser;
    addParameter(p, 'KFold',           10);
    addParameter(p, 'KernelFunction',  'rbf');
    addParameter(p, 'BoxConstraint',   2);
    addParameter(p, 'KernelScale',     11);
    addParameter(p, 'PolynomialOrder', 3);
    addParameter(p, 'Standardize',     false);
    parse(p, varargin{:});
    opt = p.Results;

    totalClassificationTimeStart = tic;

    %% ---- Prepare data ----
    % Extract feature matrices and labels for train/test sets
    [X_train, y_train, X_test, y_test, classNames] = prepareData_21(trainTable, testTable);

    fprintf('\n=== SVM | Kernel: %s | C: %g | Folds: %d ===\n', ...
            opt.KernelFunction, opt.BoxConstraint, opt.KFold);

    %% ---- K-Fold Cross-Validation on training data ----
    rng(42);
    cv = cvpartition(y_train, 'KFold', opt.KFold, 'Stratify', true);
    % cvMetricsAll = [];   % struct array to hold per-fold metrics
    cvMetricsAll_thresh = repmat(struct( ...
    'confMat', [], 'balancedAcc', [], 'accuracy', [], 'sensitivity', [], 'specificity', [], ...
    'precision', [], 'f1', [], 'mcc', [], 'kappa', [], ...
    'auc', [], 'totalInferenceTime_s', [], 'avgInferenceTime_ms', []), opt.KFold, 1);


    % Store all OOF predictions and scores for aggregate ROC
    all_yTestPred  = [];
    all_scores = [];

    for f = 1:opt.KFold

        foldClassificationTimeStart = tic;

        % Split into training and validation folds
        Xtr = X_train(cv.training(f), :);
        ytr = y_train(cv.training(f));
        Xte = X_train(cv.test(f), :);
        yte = y_train(cv.test(f));

        % Data Balancing - I : Partial undersampling
        idx_train_normal = find(ytr == -1);
        idx_train_abnormal = find(ytr == 1);
        
        nNormal_train = length(idx_train_normal);
        nAbnormal_train = length(idx_train_abnormal);
        
        if nNormal_train >= nAbnormal_train
            % Normal is majority → undersample normal
            target_majority = round(nAbnormal_train * underSamplingFactor);
            target_majority = min(target_majority, nNormal_train);
            idx_majority_sampled = idx_train_normal(randperm(nNormal_train, target_majority));
            idx_minority_sampled = idx_train_abnormal; % keep all abnormal
        else
            % Abnormal is majority → undersample abnormal
            target_majority = round(nNormal_train * underSamplingFactor);
            target_majority = min(target_majority, nAbnormal_train);
            idx_majority_sampled = idx_train_abnormal(randperm(nAbnormal_train, target_majority));
            idx_minority_sampled = idx_train_normal; % keep all normal
        end
        
        % Combine
        idx_partial_balanced = [idx_majority_sampled; idx_minority_sampled];
        idx_partial_balanced = idx_partial_balanced(randperm(length(idx_partial_balanced)));
        
        
        Xtr = Xtr(idx_partial_balanced, :);
        ytr = ytr(idx_partial_balanced);
        
     
        % Data Balancing - II : Class Weights for SVM Training
        nAbnormal = sum(ytr == 1);
        nNormal = sum(ytr == -1);
        total = length(ytr);
        
        % Compute inverse frequency weights (use either of the two below)
        
        % Option - I
        % w_normal = total / (2 * nNormal);
        % w_abnormal = total / (2 * nAbnormal);
        % w_sum = w_normal + w_abnormal;
        % w_normal = w_normal / w_sum;
        % w_abnormal = w_abnormal / w_sum;
        
        % Option - II
        w_normal = classFactor*(total / (2 * nNormal))  / ((total/(2*nNormal))  + (total/(2*nAbnormal)));
        w_abnormal = (total / (2 * nAbnormal))   / ((total/(2*nNormal))  + (total/(2*nAbnormal)));
        
        % fprintf('Computed class weights:\n');
        % fprintf('  Normal   :   %.4f\n', w_normal);
        % fprintf('  Abnormal :   %.4f\n', w_abnormal);
        
        % Cost matrix: Penalize misclassifying the minority class more
        costMatrix = [0, w_normal; w_abnormal, 0];

        % Train fold model (binary SVM)
        if strcmp(opt.KernelFunction, 'polynomial')
            mdl = fitcsvm(Xtr, ytr, ...
                'KernelFunction', opt.KernelFunction, ...
                'BoxConstraint', opt.BoxConstraint, ...
                'KernelScale', opt.KernelScale, ...
                'PolynomialOrder', opt.PolynomialOrder, ...
                'Standardize', opt.Standardize, ...
                'ClassNames', classNames, ...
                'Cost', costMatrix);
        else
            mdl = fitcsvm(Xtr, ytr, ...
                'KernelFunction', opt.KernelFunction, ...
                'BoxConstraint', opt.BoxConstraint, ...
                'KernelScale', opt.KernelScale, ...
                'Standardize', opt.Standardize, ...
                'ClassNames', classNames, ...
                'Cost', costMatrix);
        end

        % Predict labels + decision scores
        [yPredFold, scores] = predict(mdl, Xte);

        % --- Balanced Accuracy threshold tuning on validation fold ---
        thresholds = linspace(min(scores(:,2)), max(scores(:,2)), 200); % candidate thresholds
        BA = zeros(size(thresholds));
        
        for i = 1:numel(thresholds)
            preds = 2 * (scores(:,2) > thresholds(i)) - 1;  % map to {-1,+1}
            sens = sum(preds==1 & yte==1) / sum(yte==1);    % sensitivity
            spec = sum(preds==-1 & yte==-1) / sum(yte==-1); % specificity
            BA(i) = (sens + spec)/2;                        % balanced accuracy
        end
        
        [~,bestIdx] = max(BA);
        optimalThresh_fold = thresholds(bestIdx);
        foldClassificationTime = toc(foldClassificationTimeStart);
        
        % Apply tuned threshold to validation scores
        yPredFold_thresh = 2 * (scores(:,2) > optimalThresh_fold) - 1;
        
        % Compute metrics for threshold-tuned predictions
        cvMetricsAll_thresh(f) = computeMetrics_10(yte, yPredFold_thresh, classNames, ...
            sprintf('Fold-%d (BA-Tuned)', f), foldClassificationTime, numel(yte), scores);
        
        % Store threshold for later aggregation
        allThresholds(f) = optimalThresh_fold;
  
        % Store OOF predictions and scores
        yTestTable = table(yte, yPredFold, scores(:,2),...
            'VariableNames', {'TrueLabel','PredictedLabel','Score'});
        
        all_yTestPred = [all_yTestPred; yTestTable];
    
    end

    % --- Pooled Balanced Accuracy threshold selection ---
    thresholds = linspace(min(all_yTestPred.Score), max(all_yTestPred.Score), 200);
    BA = zeros(size(thresholds));
    
    for i = 1:numel(thresholds)
        preds = 2 * (all_yTestPred.Score > thresholds(i)) - 1;
        sens = sum(preds==1 & all_yTestPred.TrueLabel==1) / sum(all_yTestPred.TrueLabel==1);
        spec = sum(preds==-1 & all_yTestPred.TrueLabel==-1) / sum(all_yTestPred.TrueLabel==-1);
        BA(i) = (sens + spec)/2;
    end
    
    [~,bestIdx] = max(BA);
    finalThresh = thresholds(bestIdx);
    
    fprintf('Pooled optimal threshold (Balanced Accuracy): %.3f\n', finalThresh);

    %% ---- Aggregate across folds ----
    % Collect vectors across folds
    balAcc  = [cvMetricsAll_thresh.balancedAcc];
    acc     = [cvMetricsAll_thresh.accuracy];
    se      = [cvMetricsAll_thresh.sensitivity];
    sp      = [cvMetricsAll_thresh.specificity];
    pr      = [cvMetricsAll_thresh.precision];
    f1      = [cvMetricsAll_thresh.f1];
    mcc     = [cvMetricsAll_thresh.mcc];
    kap     = [cvMetricsAll_thresh.kappa];
    auc     = [cvMetricsAll_thresh.auc];

    % Compute mean ± std
    cvSummary.mean_balancedAcc  = mean(balAcc);cvSummary.std_balancedAcc  = std(balAcc);
    cvSummary.mean_accuracy     = mean(acc);   cvSummary.std_accuracy     = std(acc);
    cvSummary.mean_sensitivity  = mean(se);    cvSummary.std_sensitivity  = std(se);
    cvSummary.mean_specificity  = mean(sp);    cvSummary.std_specificity  = std(sp);
    cvSummary.mean_precision    = mean(pr);    cvSummary.std_precision    = std(pr);
    cvSummary.mean_f1           = mean(f1);    cvSummary.std_f1           = std(f1);
    cvSummary.mean_mcc          = mean(mcc);   cvSummary.std_mcc          = std(mcc);
    cvSummary.mean_kappa        = mean(kap);   cvSummary.std_kappa        = std(kap);
    cvSummary.mean_auc          = mean(auc);   cvSummary.std_auc          = std(auc);

    % Print summary
    fprintf('\n--- Mean ± Std across %d folds ---\n', opt.KFold);
    fprintf('BalancedAcc:  %.4f ± %.4f\n', cvSummary.mean_balancedAcc, cvSummary.std_balancedAcc);
    fprintf('Accuracy:     %.4f ± %.4f\n', cvSummary.mean_accuracy, cvSummary.std_accuracy);
    fprintf('Sensitivity:  %.4f ± %.4f\n', cvSummary.mean_sensitivity, cvSummary.std_sensitivity);
    fprintf('Specificity:  %.4f ± %.4f\n', cvSummary.mean_specificity, cvSummary.std_specificity);
    fprintf('Precision:    %.4f ± %.4f\n', cvSummary.mean_precision, cvSummary.std_precision);
    fprintf('F1-Score:     %.4f ± %.4f\n', cvSummary.mean_f1, cvSummary.std_f1);
    fprintf('MCC:          %.4f ± %.4f\n', cvSummary.mean_mcc, cvSummary.std_mcc);
    fprintf('Kappa:        %.4f ± %.4f\n', cvSummary.mean_kappa, cvSummary.std_kappa);
    fprintf('AUC:          %.4f ± %.4f\n', cvSummary.mean_auc, cvSummary.std_auc);

    %% ---- Train on full training set, evaluate on test set ----
    if strcmp(opt.KernelFunction, 'polynomial')
            fullMdl = fitcsvm(X_train, y_train, ...
                'KernelFunction', opt.KernelFunction, ...
                'BoxConstraint', opt.BoxConstraint, ...
                'KernelScale', opt.KernelScale, ...
                'PolynomialOrder', opt.PolynomialOrder, ...
                'Standardize', opt.Standardize, ...
                'ClassNames', classNames);
    else
            fullMdl = fitcsvm(X_train, y_train, ...
                'KernelFunction', opt.KernelFunction, ...
                'BoxConstraint', opt.BoxConstraint, ...
                'KernelScale', opt.KernelScale, ...
                'Standardize', opt.Standardize, ...
                'ClassNames', classNames);
    end

    [~, testScores] = predict(fullMdl, X_test);
    totalClassificationTime = toc(totalClassificationTimeStart);

    % Apply CV-derived BA threshold
    testPreds_thresh = 2 * (testScores(:,2) > finalThresh) - 1;
    testMetrics_thresh = computeMetrics_10(y_test, testPreds_thresh, classNames, ...
        'Test (BA-Tuned)', totalClassificationTime, numel(y_test), testScores);
    
    fprintf('Applied CV-derived BA threshold: %.3f\n', finalThresh);

    %% ---- Package results ----
    SVMresults.classifier           = 'SVM';
    SVMresults.kernel               = opt.KernelFunction;
    SVMresults.boxConstraint        = opt.BoxConstraint;
    SVMresults.kernelScale          = opt.KernelScale;
    SVMresults.kFold                = opt.KFold;
    SVMresults.underSamplingFactor  = underSamplingFactor;
    SVMresults.classFactor          = classFactor;
    SVMresults.all_yTestPred        = all_yTestPred;
    SVMresults.cv.folds             = cvMetricsAll_thresh;   % per-fold metrics
    SVMresults.cv.summary           = cvSummary;      % mean ± std across folds
    SVMresults.test                 = testMetrics_thresh;    % test set metrics
    SVMresults.model                = fullMdl;
    SVMresults.classNames           = classNames;


    timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    filename = ['SVM_Results_shenPresetPhysionet_' char(timestamp) '.mat'];
    save(filename, 'SVMresults');
    fprintf('\n✔ Results saved to: %s\n', filename);

end
