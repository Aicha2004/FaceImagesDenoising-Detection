import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from math import pi
from datetime import datetime

# ---------------- CONFIG ----------------
def find_metrics_csv(base_path):
    """Automatically find the metrics CSV file"""
    patterns = [
        os.path.join(base_path, "outputs", "denoised_pipeline_existing_metrics", "*.csv"),
        os.path.join(base_path, "outputs", "*_metrics", "*.csv"),
        os.path.join(base_path, "**", "denoising_metrics*.csv"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    return None

# Create output folder for graphs
def create_output_folder(base_path):
    """Create a folder to save all graphs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graphs_folder = os.path.join(base_path, "graphs_output", f"denoising_graphs_{timestamp}")
    os.makedirs(graphs_folder, exist_ok=True)
    print(f"✓ Graphs will be saved to: {graphs_folder}")
    return graphs_folder

# ---------------- LOAD DATA ----------------
base_path = r"C:\Users\DELL\Desktop\Projet_FaceDenoising"
csv_file = find_metrics_csv(base_path)

if csv_file is None:
    print("ERROR: Could not find metrics CSV file!")
    print("Searching in:", base_path)
    exit()

print(f"✓ Loading: {csv_file}")
df = pd.read_csv(csv_file)
print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")

# Create output folder
graphs_output_folder = create_output_folder(base_path)

# Display available columns
print(f"\nAvailable columns: {df.columns.tolist()}")

# ---------------- SELECT METRICS ----------------
# All available metrics in your CSV
all_metrics = ['psnr', 'ssim', 'lpips', 'f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'mcc']

# Select only metrics that exist in your CSV
metrics_to_plot = [m for m in all_metrics if m in df.columns]
print(f"\n✓ Plotting metrics: {metrics_to_plot}")

if not metrics_to_plot:
    print("No metrics found to plot!")
    exit()

# ---------------- PREPARE DATA ----------------
# Melt the dataframe for seaborn
id_vars = ['filter', 'noise_type']
df_melt = df.melt(
    id_vars=id_vars, 
    value_vars=metrics_to_plot, 
    var_name="Metric", 
    value_name="Value"
)

# Remove NaN values
df_melt = df_melt.dropna()

# Rename for better readability
df_melt = df_melt.rename(columns={'filter': 'Method'})

print(f"\nData summary:")
print(f"  - Methods: {df_melt['Method'].nunique()} unique")
print(f"  - Noise types: {df_melt['noise_type'].nunique()} unique")
print(f"  - Total combinations: {len(df_melt):,}")

# ---------------- GRAPH 1: Faceted Bar Plot ----------------
print("\n📊 Creating faceted bar plot...")
sns.set_style("whitegrid")
sns.set_palette("husl")

n_metrics = len(metrics_to_plot)
n_cols = min(3, n_metrics)

g = sns.catplot(
    data=df_melt, 
    x="Method", y="Value", hue="noise_type",
    col="Metric", kind="bar", 
    col_wrap=n_cols,
    height=4, aspect=1.5,
    sharey=False,
    legend_out=True
)

g.set_titles("{col_name}")
g.set_xticklabels(rotation=45, horizontalalignment='right', fontsize=9)
g.set_axis_labels("Denoising Method", "Metric Value")
g.fig.subplots_adjust(top=0.92)
g.fig.suptitle("Denoising Performance Comparison by Metric and Noise Type", fontsize=16, fontweight='bold')
g.add_legend(title="Noise Type", loc='center right', bbox_to_anchor=(1.12, 0.5))

plt.tight_layout()
# Save graph
faceted_plot_path = os.path.join(graphs_output_folder, "1_faceted_bar_plot.png")
plt.savefig(faceted_plot_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {faceted_plot_path}")
plt.close()

# ---------------- GRAPH 2: Individual Bar Plots ----------------
print("\n📊 Creating individual bar plots...")
for metric in metrics_to_plot:
    plt.figure(figsize=(14, 7))
    metric_data = df_melt[df_melt['Metric'] == metric]
    
    sns.barplot(
        data=metric_data, 
        x='Method', y='Value', hue='noise_type',
        palette='tab10',
        errorbar='sd'
    )
    
    better = "Higher is better" if metric not in ['lpips'] else "Lower is better"
    
    plt.title(f'{metric.upper()} - Denoising Performance Comparison\n({better})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Denoising Method', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Noise Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save graph
    individual_plot_path = os.path.join(graphs_output_folder, f"2_bar_plot_{metric}.png")
    plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {individual_plot_path}")
    plt.close()

# ---------------- GRAPH 3: Heatmap (Average Performance) ----------------
print("\n📊 Creating performance heatmaps...")
for metric in metrics_to_plot:
    metric_data = df_melt[df_melt['Metric'] == metric]
    
    pivot_table = metric_data.pivot_table(
        values='Value', 
        index='Method', 
        columns='noise_type', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(16, 8))
    cmap = 'viridis' if metric != 'lpips' else 'viridis_r'
    sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt='.3f', 
        cmap=cmap,
        cbar_kws={'label': metric.upper()},
        linewidths=0.5,
        linecolor='gray'
    )
    
    better = "Higher = Better" if metric not in ['lpips'] else "Lower = Better"
    plt.title(f'{metric.upper()} Performance Heatmap ({better})', fontsize=14, fontweight='bold')
    plt.xlabel('Noise Type', fontsize=12)
    plt.ylabel('Denoising Method', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save graph
    heatmap_path = os.path.join(graphs_output_folder, f"3_heatmap_{metric}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {heatmap_path}")
    plt.close()

# ---------------- GRAPH 4: Box Plot (Distribution) ----------------
print("\n📊 Creating distribution box plots...")
for metric in metrics_to_plot:
    plt.figure(figsize=(14, 7))
    metric_data = df_melt[df_melt['Metric'] == metric]
    
    sns.boxplot(
        data=metric_data, 
        x='Method', y='Value', hue='noise_type',
        palette='Set3'
    )
    
    plt.title(f'{metric.upper()} - Performance Distribution by Method and Noise Type', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Denoising Method', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Noise Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save graph
    boxplot_path = os.path.join(graphs_output_folder, f"4_boxplot_{metric}.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {boxplot_path}")
    plt.close()

# ---------------- GRAPH 5: Method Ranking (Bar Plot) ----------------
print("\n📊 Creating method ranking plots...")
for metric in metrics_to_plot:
    plt.figure(figsize=(12, 6))
    metric_data = df_melt[df_melt['Metric'] == metric]
    
    method_ranking = metric_data.groupby('Method')['Value'].mean().sort_values(ascending=(metric=='lpips'))
    
    method_ranking.plot(kind='barh', color='steelblue', edgecolor='black', alpha=0.7)
    
    better = "Higher is better" if metric not in ['lpips'] else "Lower is better"
    plt.title(f'{metric.upper()} - Overall Method Ranking\n({better})', fontsize=14, fontweight='bold')
    plt.xlabel(f'Average {metric.upper()}', fontsize=12)
    plt.ylabel('Denoising Method', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save graph
    ranking_path = os.path.join(graphs_output_folder, f"5_ranking_{metric}.png")
    plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {ranking_path}")
    plt.close()

# ---------------- GRAPH 6: Radar/Spider Chart for Top Methods ----------------
print("\n📊 Creating radar chart for top methods...")
# Select top 3 methods based on average performance across all metrics
method_scores = {}
for method in df_melt['Method'].unique():
    method_data = df_melt[df_melt['Method'] == method]
    avg_score = 0
    for metric in metrics_to_plot:
        metric_data = method_data[method_data['Metric'] == metric]['Value'].mean()
        if metric == 'lpips':
            metric_data = 1 - metric_data if not pd.isna(metric_data) else 0
        avg_score += metric_data if not pd.isna(metric_data) else 0
    method_scores[method] = avg_score / len(metrics_to_plot)

top_methods = sorted(method_scores, key=method_scores.get, reverse=True)[:4]

angles = [n / float(len(metrics_to_plot)) * 2 * pi for n in range(len(metrics_to_plot))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
for method in top_methods:
    values = []
    for metric in metrics_to_plot:
        metric_data = df_melt[(df_melt['Method'] == method) & (df_melt['Metric'] == metric)]['Value'].mean()
        if metric == 'lpips':
            metric_data = 1 - metric_data if not pd.isna(metric_data) else 0
        values.append(metric_data if not pd.isna(metric_data) else 0)
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=method)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.upper() for m in metrics_to_plot])
ax.set_ylim(0, 1)
ax.set_title('Method Performance Radar Chart (Normalized)', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()

# Save graph
radar_path = os.path.join(graphs_output_folder, "6_radar_chart_top_methods.png")
plt.savefig(radar_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {radar_path}")
plt.close()

# ---------------- GRAPH 7: Summary Bar Plot for Top Metrics ----------------
print("\n📊 Creating summary bar plot...")

# Select top 4 methods and average across all noise types
avg_performance = df_melt.groupby(['Method', 'Metric'])['Value'].mean().reset_index()

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics_to_plot[:4]):  # Plot first 4 metrics
    metric_data = avg_performance[avg_performance['Metric'] == metric]
    
    if metric == 'lpips':
        metric_data = metric_data.sort_values('Value', ascending=True)
    else:
        metric_data = metric_data.sort_values('Value', ascending=False)
    
    bars = axes[idx].bar(metric_data['Method'], metric_data['Value'], color='steelblue', edgecolor='black')
    axes[idx].set_ylabel(metric.upper(), fontsize=12)
    axes[idx].set_xlabel('Method', fontsize=12)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=9)

for idx in range(len(metrics_to_plot[:4]), len(axes)):
    fig.delaxes(axes[idx])

fig.suptitle('Overall Method Performance Across Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save graph
summary_path = os.path.join(graphs_output_folder, "7_summary_bar_plot.png")
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {summary_path}")
plt.close()

# ---------------- GRAPH 8: Correlation Matrix of Metrics ----------------
print("\n📊 Creating correlation matrix...")

# Select only numeric metric columns for correlation
metric_columns = [m for m in metrics_to_plot if m in df.columns]
correlation_matrix = df[metric_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt='.3f', 
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)
plt.title('Metrics Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save graph
correlation_path = os.path.join(graphs_output_folder, "8_correlation_matrix.png")
plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {correlation_path}")
plt.close()

# ---------------- PRINT SUMMARY STATISTICS ----------------
print("\n" + "="*70)
print("PERFORMANCE SUMMARY STATISTICS")
print("="*70)

for metric in metrics_to_plot:
    print(f"\n{'='*70}")
    print(f"{metric.upper()}")
    print(f"{'='*70}")
    
    metric_data = df_melt[df_melt['Metric'] == metric]
    
    if metric == 'lpips':
        best_per_noise = metric_data.loc[metric_data.groupby('noise_type')['Value'].idxmin()]
        print(f"\nBest method per noise type (Lower is better for {metric}):")
    else:
        best_per_noise = metric_data.loc[metric_data.groupby('noise_type')['Value'].idxmax()]
        print(f"\nBest method per noise type (Higher is better for {metric}):")
    
    for _, row in best_per_noise.iterrows():
        print(f"  • {row['noise_type']:20s} -> {row['Method']:15s} ({row['Value']:.4f})")
    
    if metric == 'lpips':
        overall_best = metric_data.groupby('Method')['Value'].mean().idxmin()
        overall_value = metric_data.groupby('Method')['Value'].mean().min()
        print(f"\nOverall Best Method: {overall_best} ({overall_value:.4f}) - Lower is better")
    else:
        overall_best = metric_data.groupby('Method')['Value'].mean().idxmax()
        overall_value = metric_data.groupby('Method')['Value'].mean().max()
        print(f"\nOverall Best Method: {overall_best} ({overall_value:.4f}) - Higher is better")

print("\n" + "="*70)
print("✓ All graphs generated and saved successfully!")
print(f"✓ Graphs location: {graphs_output_folder}")
print("="*70)

# Create an index.html file to easily view all graphs
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Denoising Performance Graphs</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; text-align: center; }}
        .gallery {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }}
        .graph {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px; overflow: hidden; width: 45%; }}
        .graph img {{ width: 100%; height: auto; }}
        .graph-title {{ padding: 10px; background: #f0f0f0; font-weight: bold; }}
        .summary {{ background: white; padding: 20px; margin: 20px; border-radius: 8px; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Denoising Performance Analysis Graphs</h1>
    <div class="gallery">
"""

# Add all PNG files to the HTML gallery
png_files = sorted([f for f in os.listdir(graphs_output_folder) if f.endswith('.png')])
for png_file in png_files:
    html_content += f"""
        <div class="graph">
            <div class="graph-title">{png_file.replace('_', ' ').replace('.png', '')}</div>
            <img src="{png_file}" alt="{png_file}">
        </div>
    """

html_content += """
    </div>
</body>
</html>
"""

# Save the HTML file
html_path = os.path.join(graphs_output_folder, "index.html")
with open(html_path, 'w') as f:
    f.write(html_content)
print(f"✓ Created HTML gallery: {html_path}")

print(f"\n🎉 All done! Open {html_path} in your browser to view all graphs.")