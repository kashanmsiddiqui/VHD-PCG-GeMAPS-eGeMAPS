function m = computeMetrics_11(yTrue, yPred, classNames, tag, inferenceTimeSec, nTestSamples, scores)
    
    % Confusion matrix
    yTrue = double(yTrue);
    yPred = double(yPred);
    m.confMat  = confusionmat(yTrue, yPred, 'Order', classNames);
   
    % Extract metrics
    TN = m.confMat(1,1);  % True Normal/Negative (Normal predicted as Normal)
    FP = m.confMat(1,2);  % False Positive (Normal predicted as Abnormal)
    FN = m.confMat(2,1);  % False Negative (Abnormal predicted as Normal)
    TP = m.confMat(2,2);  % True Positive (Abnormal predicted as Abnormal)

    % Calculate metrics
    m.accuracy = (TP + TN) / sum(m.confMat(:));
    m.precision = TP / (TP + FP + eps);
    m.sensitivity = TP / (TP + FN + eps);  % Abnormal detection
    m.specificity = TN / (TN + FP + eps);    % Normal detection
    m.f1 = 2 * m.precision * m.sensitivity / (m.precision + m.sensitivity + eps);
    m.balancedAcc = (m.sensitivity + m.specificity) / 2;  

    % Matthew's Correlation Coefficient (MCC)
    m.mcc = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + eps);

    % Kappa
    N = sum(m.confMat(:));
    p_o = (TP + TN) / N;                                            % Observed agreement
    p_e = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (N^2);  % Expected agreement
    m.kappa = (p_o - p_e) / (1 - p_e + eps);

    % AUC (binary only, requires scores)
    if nargin>=7 && ~isempty(scores)
        [X,Y,T,AUC] = perfcurve(yTrue, scores(:,2), classNames(2));
        m.auc = AUC;

        % Optional: store ROC curve data
        m.rocX = X;   % False Positive Rate
        m.rocY = Y;   % True Positive Rate
        m.rocT = T;   % Thresholds
    
        % Plot ROC curve
        figure;
        plot(X, Y, 'b-', 'LineWidth', 2);
        hold on;
        plot([0 1], [0 1], 'k--'); % diagonal reference
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title(sprintf('ROC Curve [%s] (AUC = %.3f)', tag, AUC));
        grid on;

    else
        m.auc = NaN;
    end

    % Inference time
    if nargin>=5 && ~isempty(inferenceTimeSec) && nargin>=6 && nTestSamples>0
        m.totalInferenceTime_s = inferenceTimeSec;
        m.avgInferenceTime_ms  = (inferenceTimeSec/nTestSamples)*1000;
    else
        m.totalInferenceTime_s = NaN; 
        m.avgInferenceTime_ms = NaN;
    end

    % Print summary
    fprintf('  [%s] BalAcc: %.4f | Acc: %.4f | Sens: %.4f | Spec: %.4f | Prec: %.4f | F1: %.4f | MCC: %.4f | Kappa: %.4f | AUC: %.4f | AvgInf: %.3f ms/sample\n', ...
            tag, m.balancedAcc, m.accuracy, m.sensitivity, m.specificity, m.precision, ...
            m.f1, m.mcc, m.kappa, m.auc, m.avgInferenceTime_ms);
end
