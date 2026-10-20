function results = classifySVM_23(trainTable, testTable, varargin)
% classifySVM
% SVM classification with k-fold cross-validation on training data,
% then evaluation on test data.
%
% INPUTS:
%   trainTable : feature table with FileName, Label, feature columns
%   testTable  : feature table with FileName, Label, feature columns
%
% OPTIONAL NAME-VALUE INPUTS:
%   'KFold'             : number of CV folds                (default: 5)
%   'KernelFunction'    : 'rbf', 'linear', 'polynomial'     (default: 'rbf')
%   'BoxConstraint'     : C [0.1,1,10,100]                  (default: 1)
%   'KernelScale'       : Gamma (in 'rbf') [0.01,0.1,1,10]  (default: 'auto')
%   'PolynomialOrder'   : degree (in 'polynomial') [2,3,4]  (default: 3)
%   'Standardize'       : true/false                        (default: false)
%
% OUTPUT:
%   results : struct with CV (per-fold + summary) and test metrics
%
% EXAMPLE:
%   results = classifySVM_13(trainData, testData)
%   results = classifySVM(trainTable, testTable, 'KFold', 10, 'KernelFunction', 'linear');

    %% ---- Tuning Parameters ----
    % Under-sampling Factor for unbalanced dataset
    underSamplingFactor = 1.9;    % value of 1.5 means 60%-40%
    classFactor = 1;              % affects normal class

    %% ---- Parse inputs ----
    p = inputParser;
    addParameter(p, 'KFold',           5);
    addParameter(p, 'KernelFunction',  'rbf');
    addParameter(p, 'BoxConstraint',   1);
    addParameter(p, 'KernelScale',     'auto');
    addParameter(p, 'PolynomialOrder', 3);
    addParameter(p, 'Standardize',     false);
    parse(p, varargin{:});
    opt = p.Results;

    %% ---- Prepare data ----
    % Extract feature matrices and labels for train/test sets
    [X_train, y_train, X_test, y_test, classNames] = prepareData_22(trainTable, testTable);

    fprintf('\n=== SVM | Kernel: %s | C: %g | Folds: %d ===\n', ...
            opt.KernelFunction, opt.BoxConstraint, opt.KFold);

    %% ---- K-Fold Cross-Validation on training data ----
    rng(42);
    cv = cvpartition(y_train, 'KFold', opt.KFold, 'Stratify', true);
    % cvMetricsAll = [];   % struct array to hold per-fold metrics
    cvMetricsAll = repmat(struct( ...
    'confMat', [], 'balancedAcc', [], 'accuracy', [], 'sensitivity', [], 'specificity', [], ...
    'precision', [], 'f1', [], 'mcc', [], 'kappa', [], ...
    'auc', [], 'totalInferenceTime_s', [], 'avgInferenceTime_ms', []), opt.KFold, 1);


    for f = 1:opt.KFold
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
        tic;
        [yPredFold, scores] = predict(mdl, Xte);
        inferenceTimeSec = toc;

        % Compute metrics for this fold
        cvMetricsAll(f) = computeMetrics_10(yte, yPredFold, classNames, ...
                                         sprintf('Fold-%d', f), inferenceTimeSec, numel(yte), scores);
    end

    %% ---- Aggregate across folds ----
    % Collect vectors across folds
    balAcc  = [cvMetricsAll.balancedAcc];
    acc     = [cvMetricsAll.accuracy];
    se      = [cvMetricsAll.sensitivity];
    sp      = [cvMetricsAll.specificity];
    pr      = [cvMetricsAll.precision];
    f1      = [cvMetricsAll.f1];
    mcc     = [cvMetricsAll.mcc];
    kap     = [cvMetricsAll.kappa];
    auc     = [cvMetricsAll.auc];

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

    tic;
    [testPreds, testScores] = predict(fullMdl, X_test);
    inferenceTimeSec_test = toc;

    testMetrics = computeMetrics_11(y_test, testPreds, classNames, 'Test', ...
                    inferenceTimeSec_test, numel(y_test), testScores);

    %% --- Threshold tuning ---
    [X,Y,T,AUC] = perfcurve(y_test, testScores(:,2), 1);
    [~,optIdx] = max(Y - X);
    optimalThresh = T(optIdx);
    
    testPreds_thresh = testScores(:,2) > optimalThresh;
    % Convert logicals (0/1) into -1/+1 labels
    testPreds_thresh = ones(size(y_test));          % default abnormal (+1)
    testPreds_thresh(testScores(:,2) <= optimalThresh) = -1;  % assign normal (-1)
    
    testMetrics_thresh = computeMetrics_11(y_test, testPreds_thresh, [-1 1], ...
        'Test (Threshold-Tuned)', inferenceTimeSec_test, numel(y_test), testScores);
    
    fprintf('Optimal threshold (Youden J): %.3f\n', optimalThresh);

    %% ---- Package results ----
   results.classifier     = 'SVM';
    results.kernel         = opt.KernelFunction;
    results.boxConstraint  = opt.BoxConstraint;
    results.kernelScale    = opt.KernelScale;
    results.kFold          = opt.KFold;
    results.cv.folds       = cvMetricsAll;   % per-fold metrics
    results.cv.summary     = cvSummary;      % mean ± std across folds

    % Default test evaluation
    results.test           = testMetrics;    % metrics at default threshold
    results.test.scores    = testScores;     % raw decision scores
    results.test.yTrue     = y_test;         % ground truth labels
    results.test.yPred     = testPreds;      % predicted labels (default threshold)
    
    % Threshold-tuned test evaluation
    results.test_tuned           = testMetrics_thresh; % metrics at tuned threshold
    results.test_tuned.scores    = testScores;         % same scores
    results.test_tuned.yTrue     = y_test;             % same ground truth
    results.test_tuned.yPred     = testPreds_thresh;   % predicted labels (tuned threshold)
    
    % Threshold info
    results.threshold.optimal    = optimalThresh;
    results.threshold.method     = 'Youden J';
    results.threshold.default    = 0.5;
    
    % Model + metadata
    results.model          = fullMdl;
    results.classNames     = classNames;

    % timestamp = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    % filename = ['SVM_Results_shenPhysionet_8020AllFolders' char(timestamp) '.mat'];
    % save(filename, 'results');
    % fprintf('\n✔ Results saved to: %s\n', filename);

end
