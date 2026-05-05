T = readtable(fullfile('results', 'robustness_metrics.csv'));

methods = string(T.Method);
methods = replace(methods, "blur_", "blur ");
methods = replace(methods, "compressed_", "comp ");
methods = replace(methods, "low_light_", "low ");

colors = [
    0.05 0.20 0.55;   % ACC dark blue
    0.75 0.15 0.05;   % F1 dark red
    0.90 0.65 0.00;   % AUC dark gold
    0.35 0.10 0.50    % MCC dark purple
];

fig = figure;
set(fig, 'Position', [100 100 1500 650]);

b = bar(categorical(methods), [T.ACC T.F1 T.AUC T.MCC], 'grouped');

for i = 1:length(b)
    b(i).FaceColor = colors(i,:);
end


set(gca, ...
    'FontSize',     19, ...
    'FontWeight', 'bold', ...
    'XTickLabelRotation', 35, ...
    'LineWidth', 1.3, ...
    'Box', 'on');

ylim([0 1.05]);
grid on;

legend({'ACC','F1','AUC','MCC'}, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'FontSize', 19, ...
    'FontWeight', 'bold');

exportgraphics(fig, fullfile('results', 'robustness_classification.png'), ...
    'Resolution', 600);

exportgraphics(fig, fullfile('results', 'robustness_classification.pdf'), ...
    'ContentType', 'vector');

close(fig);

disp('Robustness classification plot saved.');