import pandas as pd
from jiwer import cer
import matplotlib.pyplot as plt
import seaborn as sns # for figures 
from scipy.stats import shapiro, wilcoxon, mannwhitneyu
import scipy.stats as stats


def calculate_cer(data, gold_column, transcription_columns):
    cer_results = {}
    for col in transcription_columns:
        cer_results[col] = [cer(data[gold_column][i], data[col][i]) for i in range(len(data))]
    return cer_results


def calculate_mean_cer(cer_df, transcription_columns):
    cer_df['HSR'] = cer_df[transcription_columns].mean(axis=1)
    columns_order = ['WHISPER', 'HSR'] + transcription_columns
    return cer_df[columns_order]


# Load your data
data = pd.read_csv('./Transcriptions.csv')

# Specify the columns
gold_column = 'Gold answer'
transcription_columns = ['Participant 1', 'Participant 2', 'Participant 3', 'Participant 4', 'Participant 5', 'Participant 6']

# Calculate CER
cer_results = calculate_cer(data, gold_column, ['WHISPER'] + transcription_columns)

# Convert results to a DataFrame  
cer_df = pd.DataFrame(cer_results)

# Calculate mean CER for participants and save CER results for later processing
cer_df = calculate_mean_cer(cer_df, transcription_columns)
cer_df.to_csv('./CER_Results.csv')
print(cer_df)

# Basic statistics of data 
whisper_mean = cer_df["WHISPER"].mean()
whisper_std = cer_df["WHISPER"].std()
mean_participants_mean = cer_df["HSR"].mean()
mean_participants_std = cer_df["HSR"].std()
print(f"Mean of WHISPER： {whisper_mean}, Std of WHISPER {whisper_std}\nMean of HSR： {mean_participants_mean}, Std of HSR： {mean_participants_std}")


# List all available style to choose from
# print(plt.style.available)
# plt.style.use('seaborn-v0_8-darkgrid')

# Figure 1
# Create the boxplot of WHISPER and all participants
plt.figure(figsize=(12, 6))
sns.boxplot(data=cer_df, palette="pastel", linewidth=2)
plt.ylabel("CER")
plt.grid(ls='--', axis='y')
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("CER", fontsize=15)
plt.show()  # Figure 1

# Figure 2
# Visualization - Boxplot and Paired Data Plot
plt.figure(figsize=(10, 8))
sns.boxplot(data=cer_df[['WHISPER', 'HSR']], palette="pastel", width=0.5,  linewidth=2)
sns.swarmplot(data=cer_df[['WHISPER', 'HSR']], color="dimgray", alpha=0.6)
for i in range(len(cer_df)):
    plt.plot([0, 1], [cer_df['WHISPER'][i], cer_df['HSR'][i]], color='grey', alpha=0.5)

plt.grid(ls='--', axis='y')
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.ylabel("CER", fontsize=15)
plt.xticks([0, 1], ['WHISPER', 'HSR'], fontsize=15)
plt.yticks(fontsize=15)
plt.show()  # Figure 2


# Figure 3
# Preliminary Normality Test and Visualize the Normality
for column in ['WHISPER', 'HSR']:
    stat, p = shapiro(cer_df[column])
    print(f'Normality test for {column}: Statistics={stat}, p={p}')
# Function to create Q-Q plot


def qqplot(data, title, line_color, scatter_color):
    plt.figure(figsize=(6, 9))
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=plt)
    plt.scatter(osm, osr, color=scatter_color)  # Change scatter points color
    plt.plot(osm, slope*osm + intercept, color=line_color)  # Change line color    
    # plt.title(title)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()


# Q-Q plot for WHISPER
qqplot(cer_df['WHISPER'], 'Q-Q Plot of WHISPER CER', line_color='skyblue', scatter_color='darkblue')
# Q-Q plot for HSR
qqplot(cer_df['HSR'], 'Q-Q Plot of HSR CER', line_color='skyblue', scatter_color='darkblue')

### Statisctical tests
# Perform Wilcoxon Signed-Rank Test on paired data (WHISPER and HSR)
wilcoxon_stat, wilcoxon_p = wilcoxon(cer_df['WHISPER'], cer_df['HSR'])
print(f'Wilcoxon Signed-Rank Test:\nStatistics={wilcoxon_stat}, p-value={wilcoxon_p}\n')

# Perform Mann-Whitney U Test on the independent samples (First 8 vs Last 8 sentences)
# Split the data into two groups for both WHISPER and HSR
first_8_cer_whisper = cer_df.iloc[:8]['WHISPER']
last_8_cer_whisper = cer_df.iloc[8:]['WHISPER']
first_8_cer_hsr = cer_df.iloc[:8]['HSR']
last_8_cer_hsr = cer_df.iloc[8:]['HSR']

# Mann-Whitney U Test for WHISPER
mannwhitney_stat_whisper, mannwhitney_p_whisper = mannwhitneyu(first_8_cer_whisper, last_8_cer_whisper)
# Mann-Whitney U Test for HSR
mannwhitney_stat_hsr, mannwhitney_p_hsr = mannwhitneyu(first_8_cer_hsr, last_8_cer_hsr)
print(f'Mann-Whitney U Test for WHISPER:\nStatistics={mannwhitney_stat_whisper}, p-value={mannwhitney_p_whisper}\n')
print(f'Mann-Whitney U Test for HSR:\nStatistics={mannwhitney_stat_hsr}, p-value={mannwhitney_p_hsr}\n')

# Figure 4: CER Comparison: First 8 vs Last 8 Sentences for WHISPER and HSR
# Create a combined DataFrame for plotting
cer_data_for_plotting = pd.DataFrame({
    'First 8 - WHISPER': first_8_cer_whisper,
    'Last 8 - WHISPER': last_8_cer_whisper,
    'First 8 - HSR': first_8_cer_hsr,
    'Last 8 - HSR': last_8_cer_hsr
})

plt.figure(figsize=(10, 8))
sns.boxplot(data=cer_data_for_plotting, palette="pastel", linewidth=2)
plt.xticks(range(4), ['First 8 - WHISPER', 'Last 8 - WHISPER', 'First 8 - HSR', 'Last 8 - HSR'], fontsize=13)
plt.grid(ls='--', axis='y')
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.yticks(fontsize=12)
plt.ylabel("CER", fontsize=15)
plt.show()