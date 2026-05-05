resultsPath = 'results/metrics_results.csv';
T = readtable(resultsPath);

shortNames = strings(height(T),1);

for i = 1:height(T)
    name = lower(T.Method{i});

    if contains(name,'adaptive')
        shortNames(i) = "adap";
    elseif contains(name,'gaussian')
        shortNames(i) = "gauss";
    elseif contains(name,'median')
        shortNames(i) = "median";
    elseif contains(name,'max')
        shortNames(i) = "max";
    elseif contains(name,'min')
        shortNames(i) = "min";
    else
        shortNames(i) = name;
    end
end

colors = [
    0.05 0.20 0.55;   % ACC dark blue
    0.75 0.15 0.05;   % PRE dark red
    0.90 0.65 0.00;   % REC dark gold
    0.35 0.10 0.50;   % F1 dark purple
    0.10 0.45 0.10;   % AUC dark green
    0.00 0.45 0.55    % MCC dark cyan
];

fig = figure;
set(fig,'Position',[100 100 1500 650]);

b = bar(categorical(shortNames), ...
    [T.ACC T.PRE T.REC T.F1 T.AUC T.MCC], ...
    'grouped');

for i = 1:length(b)
    b(i).FaceColor = colors(i,:);
end



set(gca, ...
    'FontSize',19, ...
    'FontWeight','bold', ...
    'XTickLabelRotation',25, ...
    'LineWidth',1.3, ...
    'Box','on');

ylim([0 1.05]);
grid on;

legend({'ACC','PRE','REC','F1','AUC','MCC'}, ...
    'Location','northoutside', ...
    'Orientation','horizontal', ...
    'FontSize',19, ...
    'FontWeight','bold');

exportgraphics(fig,'results/metrics_comparison.png','Resolution',600);

exportgraphics(fig,'results/metrics_comparison.pdf', ...
    'ContentType','vector');

close(fig);

disp('Improved comparison plot saved.');