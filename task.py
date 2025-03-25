# %% [markdown]
# **Import libraries**
# 
# This section imports the necessary libraries for data manipulation and analysis.

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pychattr.channel_attribution import MarkovModel
from pychattr.channel_attribution import HeuristicModel
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline

# %% [markdown]
# **Read the CSV and Inspect the Data**

# %%
df = pd.read_csv('TBA_attribution_data.csv')

# %%
# Inspect the structure
print(df.head())

# %%
#Print (num of rows, columns)
print(df.shape)

# %%
#Print columns and their data types
print(df.dtypes)
print(df.info()) 

# %% [markdown]
# Time Frame of Data, checking from when to when the rows are

# %%
min_time = df['time'].min()
max_time = df['time'].max()
print("Minimum time:", min_time)
print("Maximum time:", max_time)

# %% [markdown]
# Check columns values as in GROUP BY in SQL

# %%
for col in ['interaction', 'conversion', 'channel']:  
    print(f"Value counts for {col}:")
    print(df[col].value_counts())
    print("----------------")

# %% [markdown]
# **First Data Pre-processing task** : Determine Data types for Columns and see if anything needs to be changed.

# %%
df.dtypes

# %% [markdown]
# Given that the 'time' column seems to be carrying timestamps I would convert it to a datetime object. 
# This will allow us to perform operations on the time column, like compare which timestamp is before and after each other. The current data type 'object' is more for strings than timestamps although there

# %%
print('before transformation \'time\' column was:', df['time'].dtypes)
df['time']=pd.to_datetime(df['time'])
#print(df['time'].tail())
print('after transdormation the type of time column:',df['time'].dtypes)

# %% [markdown]
# Check for NULLs

# %%
null_counts = df.isnull().sum()
print(null_counts)

# %% [markdown]
# Show rows with NULL. There are 9 in conversion column so let's show these ones.

# %%
null_rows = df[df['conversion'].isnull()]
print(null_rows)

# %% [markdown]
# Locate the rows with 0 and if the conversion value is 0 then it makes sense to fill the NULL in conversion to 0 as no value /money means no conversion happened. 
# We see that already from the above but for demonstration purposes we use the AND operator because in real life it might be 1.

# %%
null_rows = df.loc[(df['conversion'].isnull()) & (df['conversion_value'] == 0)]

# %% [markdown]
# Locate and change WHEN conversion is null AND when conversion_value=0 then make Conversion to 0

# %%
#checks if the conversion column is null and the conversion_value column is 0 then it will replace the null value with 0
df.loc[(df['conversion'].isnull()) & (df['conversion_value'] == 0), 'conversion'] = 0

null_counts = df.isnull().sum()
print(null_counts)

# %% [markdown]
# **Check for Duplicates first in across ALL columns then a subset**

# %%
duplicate_rows = df[df.duplicated(keep=False)]

duplicate_rows = duplicate_rows.sort_values(by='time')

print(duplicate_rows)

num_duplicates = df.duplicated().sum()

print("\nNumber of Duplicate Rows:", num_duplicates)

# %% [markdown]
# 4345 duplicates found

# %% [markdown]
# Remove the duplicates

# %%
df.drop_duplicates(inplace=True)

print("\nDuplicate Rows after removing duplicates:",df[df.duplicated()].shape)

# %% [markdown]
# Rows that are duplicate couple with 1 conversion and 1 impresion and dropping the 'impression' one in favour of the conversion

# %%
duplicate_rows = df[df.duplicated(subset=['cookie', 'time'], keep=False)]

# Sort the duplicate rows by time
duplicate_rows = duplicate_rows.sort_values(by='time')

# Filter duplicate_rows where 'interaction' column is 'conversion'
print("all dup row based on cookie and time", len(duplicate_rows))
impression_duplicates = duplicate_rows[duplicate_rows['interaction'] == 'impression']

conversion_duplicates = duplicate_rows[duplicate_rows['interaction'] == 'conversion']
#print(conversion_duplicates[['time','cookie']])

# Merge and keep all columns from impression duplicates, only 'name' from conversion_duplicates
merged_df = impression_duplicates.merge(conversion_duplicates[['cookie', 'time']], on=['cookie', 'time'], how='inner')

# Print the result
print("all duplicate row based on cookie and time that are impressions",len(impression_duplicates))
print(len(conversion_duplicates))
# Print the duplicate rows
#conversion_rows=duplicate_rows[duplicate_rows['interaction']=='conversion']
#print(len(conversion_rows['interaction']=='conversion'))

#finds duplicates with same cookie and time in the intial dataframe that have an interaction with 'impression' and removes them
print("before drop",len(df))
df = df.merge(merged_df, how='left', indicator=True)
df = df[df['_merge'] == 'left_only'].drop(columns=['_merge'])
print("after drop",len(df))


duplicate_rows = df[df.duplicated(subset=['cookie', 'time'], keep=False)]
len(duplicate_rows)


# %% [markdown]
# After removing the ones that have an 'impression' and same timestamp with conversion let's proceed to remove the ones that have different channel.
# Given the popularity of 'Facebook' as a channel I am taking this as a standard value and removing the other ones like 'Instagram'

# %% [markdown]
# Code belows proves that Facebook is the second most frequent channel so it will take precedence over others.

# %%
# Filter for rows where interaction is 'impression'
impression_rows = df[df['interaction'] == 'impression']

# Group by channel and count unique cookies
channel_cookie_counts = impression_rows.groupby('channel')['cookie'].nunique()

# Print the channel cookie counts
print("Number of unique cookies per channel where interaction is 'impression':")
print(channel_cookie_counts)

# %%
# Find duplicate rows based on cookie and time
duplicate_rows = df[df.duplicated(subset=['cookie', 'time'], keep=False)]

# Group by cookie and time, then aggregate channels into lists
channel_couples = duplicate_rows.groupby(['cookie', 'time'])['channel'].apply(list).reset_index(name='channels')

# Count the occurrences of each channel couple
channel_couple_counts = channel_couples['channels'].value_counts()

# Print the channel couple counts
print("\nOccurrences of each channel couple:")
print(channel_couple_counts)
#Print all that are not Facebook or Paid Search
print(duplicate_rows[(duplicate_rows['channel'] != 'Facebook') & (duplicate_rows['channel'] != 'Paid Search')][['cookie', 'time', 'channel']])
# Find duplicate rows based on cookie and time such that the channel is not 'Facebook' or 'Paid Search'
to_be_removed=duplicate_rows[(duplicate_rows['channel'] != 'Facebook') & (duplicate_rows['channel'] != 'Paid Search')][['cookie', 'time', 'channel']]


# %% [markdown]
# Doing a left join adding an indicator _merged that will get a left_only for the ones matched in df and 'both' when it join based on same cookie, time and channel with the to_be_removed dataframe which is basically everything not facebook or paid search 

# %%

print('duplicates before drop:',len(df[df.duplicated(subset=['cookie', 'time'], keep=False)]))

# Merge the to_be_removed DataFrame with the original DataFrame , a LEFT JOIN
df=df.merge(to_be_removed, how='left', on=['cookie','time','channel'], indicator=True)

print("before drop",len(df))
#print(df[df['_merge'] == 'both'])

# Drop rows where the merge indicator is both meaning a match was found
df = df.drop(df[df['_merge'] == 'both'].index)

# Drop the merge indicator column
df = df.drop(columns=['_merge'])

print("after drop",len(df))


# %% [markdown]
# Final check on duplicates

# %%
print(len(df))
print("duplicates after drop:",len(df[df.duplicated(subset=['cookie', 'time'], keep=False)]))

# %% [markdown]
# Detect the Outliers via the Interquartile Range Method Rule

# %%
# Filter for rows where interaction is 'conversion', we want to exclude the ones with 0 as we want to check for outliers in terms of value of purchase
df_filtered = df[df['conversion_value'] != 0]

# Calculate Q1 and Q3 on the filtered data 
#Q1 is the value below which 25% of the data falls
Q1 = df_filtered['conversion_value'].quantile(0.25)
#Q3 is the value below which 75% of the data falls
Q3 = df_filtered['conversion_value'].quantile(0.75)
#Interquartile range
IQR = Q3 - Q1

# Define the multiplier for outliers
multiplier = 1.5

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - multiplier * IQR
upper_bound = Q3 + multiplier * IQR

# Identify outliers
# Filter for rows where conversion_value is an outlier (below lower bound or above upper bound)
outliers = df_filtered[(df_filtered['conversion_value'] < lower_bound) | (df_filtered['conversion_value'] > upper_bound)]

# Print the outliers
print("Q1 (excluding 0 values):", Q1)
print("Q3 (excluding 0 values):", Q3)
print("Lower bound (excluding 0 values):", lower_bound)
print("Upper bound (excluding 0 values):", upper_bound)
print("IQR (excluding 0 values):", IQR)
print("Outliers (excluding 0 values):")
print(outliers)

# Print the number of outliers
print("\nNumber of Outliers:", len(outliers))
len(df)

# %% [markdown]
# Remove the outliers

# %%
# Remove outliers 
df = df.drop(outliers.index)

# Display the shape of the DataFrame after removing outliers
print("Shape of DataFrame after removing outliers:", df.shape)
len(df)
df.info()

# %% [markdown]
# Show aggregate functions per channel

# %%
# Grouping by channel and counting the number of conversions
channel_stats = df[df['conversion'] == 1].groupby('channel')['conversion_value'].agg(['min', 'max', 'mean', 'median', 'count'])

# Print the results
print(channel_stats)

# Save the number of touchpoints per channel for conversion rate later on in the code
touchpoints = df.groupby('channel')['interaction'].agg('count')

# Rename the column to 'touchpoints'
touchpoints.name = 'touchpoints'

# Display the touchpoints
print(touchpoints)


# %% [markdown]
# Chart to plot time series revenue per daily

# %%
print('conversions per', df.groupby('channel')['conversion'].agg('sum'))
total_sessions = df['cookie'].nunique()
print(f'Total sessions: {total_sessions}')
total_revenue = df['conversion_value'].sum()
print(f'Total revenue: {total_revenue}')
# Ensure the 'time' column is in datetime format
df['time'] = pd.to_datetime(df['time'])

# Group by date (ignoring the time component) and calculate the total revenue per day
daily_revenue = df.groupby(df['time'].dt.date)['conversion_value'].agg('sum')

# Remove the last day from the data
daily_revenue = daily_revenue.iloc[:-1]

# Smooth the line using spline interpolation
x = np.arange(len(daily_revenue))  # Create an array of indices for the x-axis
y = daily_revenue.values  # Get the revenue values
x_smooth = np.linspace(x.min(), x.max(), 300)  # Create a smoother x-axis
y_smooth = make_interp_spline(x, y)(x_smooth)  # Interpolate the y-axis values

# Plot the smoothed line
plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, color='#41042C', linewidth=2)  # Smooth line without markers

# Format x-axis labels to show day of the month and day of the week
formatted_labels = [date.strftime('%d %a') for date in daily_revenue.index]  # Format as 'day of month day of week'
plt.xticks(ticks=np.arange(len(daily_revenue)), labels=formatted_labels, rotation=45)

# Add titles and labels
plt.title('Daily Revenue', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Revenue', fontsize=12)

# Add grid for better readability
plt.grid(True)

# Adjust layout for clarity
plt.tight_layout()

# Display the plot
plt.show()

# Print the daily revenue
print("Revenue per day", daily_revenue)

cleanimpressions=print("All interactions per channel",df.groupby('channel')['interaction'].agg('count'))

# %% [markdown]
# Chart to demonstrate which channel has the most touchpoints (impressions + conversions)

# %%
# Group by 'channel' and count the number of interactions
cleanimpressions = df.groupby('channel')['interaction'].agg('count').reset_index()
cleanimpressions.columns = ['channel', 'touchpoints']  # Rename columns for clarity

# Sort the DataFrame by the 'impressions' column in ascending order
cleanimpressions = cleanimpressions.sort_values(by='touchpoints', ascending=True)

# HEX codes for colours from TBAuction website
colors = ['#41042C', '#9baf7e', '#357b49', '#0073AA','#170B53']

# Plot cleanimpressions as a bar chart
ax = cleanimpressions.plot(
    x='channel',  # Use 'channel' as the x-axis
    y='touchpoints',  # Use 'impressions' as the y-axis
    kind='bar', 
    figsize=(10, 8), 
    color=colors,  # Set a custom color for the bars
    width=0.8,
    legend=False  # Hide the legend since there's only one bar type
)

# Customize x-axis labels and rotation
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

# Add titles and axis labels
plt.title('Touchpoints per Channel', fontsize=16)
plt.xlabel('Channel', fontsize=12)
plt.ylabel('Number of Touchpoints', fontsize=12)

# Automatically add data labels to the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9)

# Adjust the y-axis range to add more space above the bars
max_y = cleanimpressions['touchpoints'].max()  # Find the maximum value in the 'impressions' column
plt.ylim(0, max_y * 1.1)  # Add 10% extra space above the tallest bar

# Adjust layout for clarity
plt.tight_layout()

# Display the plot
plt.show()

# %% [markdown]
# Transform the data frame to become a Path Oriented data frame to use for attribution modelling and markov chains
# following the guidelines in this package : https://github.com/jmwoloso/pychattr
#     "path": [
#         "A >>> B >>> A >>> B >>> B >>> A",
#         "A >>> B >>> B >>> A >>> A",
#         "A >>> A"
#     ],
#     "conversions": [1, 1, 1],
#     "revenue": [1, 1, 1],
#     "cost": [1, 1, 1]
# 
#     There is no cost given so we will not use that one for our further analysis. Although that would be excellent for ROI calculation.

# %%
# Step 1: Sort by cookie and time to maintain sequence
df = df.sort_values(['cookie', 'time'])

# Step 2: Group by cookie
result = df.groupby('cookie').agg({
    # add >>> between channels as delimiter because this is a prerequiste for the library we are going to use later on 
    'channel': lambda x: ' >>> '.join(x),
    # will grab 1 if there is a conversion and 0 if there is not
    'conversion': 'max',
    # calculate the sum of conversion values = revenue
    'conversion_value': 'sum'
}).reset_index()


# Step 3: Rename columns => channel to path and conversion to conversions and conversion_value to revenue
result.rename(columns={'channel': 'path', 'conversion': 'conversions', 'conversion_value': 'revenue'}, inplace=True)

# Step 4: Drop cookie column if not needed
result = result[['path', 'conversions', 'revenue']]
# Change the data type of the conversions column to integer
result['conversions'] = result['conversions'].astype(int)
# Filter for paths with at least one conversion
pathdf=result[result['conversions']>0]
# Show the number of paths
len(pathdf)

# %% [markdown]
# Preparing the model from pychattr following the guidelines in this package : https://github.com/jmwoloso/pychattr

# %%
path_feature="path"
conversion_feature="conversions"
null_feature=None
revenue_feature="revenue"
cost_feature="cost"
separator=">>>"
k_order=1
n_simulations=10000
max_steps=None
return_transition_probs=True
random_state=26

# instantiate the model
mm = MarkovModel(path_feature=path_feature,
                 conversion_feature=conversion_feature,
                 null_feature=null_feature,
                 revenue_feature=revenue_feature,
                 cost_feature=cost_feature,
                 separator=separator,
                 k_order=k_order,
                 n_simulations=n_simulations,
                 max_steps=max_steps,
                 return_transition_probs=return_transition_probs,
                 random_state=random_state)

# fit the model
mm.fit(pathdf)

# %%
mm.attribution_model_
mm.attribution_model_.rename(columns={
    'channel_name': 'channel',
    'total_conversions': 'markov_conversions',
    'total_revenue': 'markov_revenue'
}, inplace=True)


# %%
mm.attribution_model_

# %%
# view the transition matrix
#sorted_df = mm.transition_matrix_.sort_values(by='transition_probability', ascending=False)
#print(sorted_df)


# %%
# view the removal effects
#print(mm.removal_effects_)

# %% [markdown]
# Preparing the data for other models for first, last and linear touch

# %%
path_feature="path"
conversion_feature="conversions"
null_feature=None
revenue_feature="revenue"
separator=">>>"
first_touch=True
last_touch=True
linear_touch=True

# instantiate the model
hm = HeuristicModel(path_feature=path_feature,
                    conversion_feature=conversion_feature,
                    null_feature=null_feature,
                    revenue_feature=revenue_feature,
                    separator=separator,
                    first_touch=first_touch,
                    last_touch=last_touch,
                    linear_touch=linear_touch,)

# fit the model
pathdf = pathdf.reset_index(drop=True)
hm.fit(pathdf)

# %%
hm.attribution_model_

# %% [markdown]
# 

# %% [markdown]
# Dropping the ensemble model

# %%
newhm=hm.attribution_model_.drop(columns=['ensemble_conversions','ensemble_revenue'])
newhm

# %% [markdown]
# Join Markov Chain result with First/Last/Linear mode to create one data frame and also adding 

# %%
#left join Heuristic model with Markov model
att_model_comb=newhm.merge(mm.attribution_model_, on='channel', how='left')
print(att_model_comb)


conversion_rates=att_model_comb.merge(touchpoints, on='channel', how='left')

# Build the list of columns to divide, excluding 'touchpoints' and 'channel'
columns_to_divide = [col for col in conversion_rates.columns if col not in ['touchpoints', 'channel']]

# For each column in columns_to_divide, create a new column with '_rate' appended to the name
for col in columns_to_divide:
    conversion_rates[f"{col}_rate"] = conversion_rates[col] / conversion_rates['touchpoints']

# Drop the original columns except 'channel' and 'touchpoints'
conversion_rates = conversion_rates.drop(columns=columns_to_divide + ['touchpoints'])
print(conversion_rates)

# %% [markdown]
# **Plotting based on Attribution method**

# %%
# Plot Conversions per Channel

# HEX codes for colours from TBAuction website
colors = ['#41042C', '#9baf7e', '#357b49', '#0073AA']

# Define the columns to plot and their corresponding display names
conversions = ['first_touch_conversions', 'last_touch_conversions', 'linear_touch_conversions', 'markov_conversions']
conversion_labels = ['First Touch', 'Last Touch', 'Linear Touch', 'Markov Chain']  # Custom labels for the legend

# Create the grouped bar plot
ax = att_model_comb[conversions].plot(
    kind='bar', 
    figsize=(10, 8), 
    width=0.8,
    color=colors
)

# Customize x-axis labels and rotation
plt.xticks(range(len(att_model_comb)), att_model_comb['channel'], rotation=45)

# Add titles and axis labels
plt.title('Conversions per Channel')
plt.ylabel('Conversions')
plt.xlabel('Channel')

# Add a legend with custom labels
plt.legend(conversion_labels, title="Attribution Type")

# Automatically add data labels to each bar
for container in ax.containers:   
    ax.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9)  # fmt='%.0f' => formats the data labels as integers (no decimal places)

# Adjust the y-axis range to add more space above the bars
max_y = max([bar.get_height() for container in ax.containers for bar in container])  # Find the tallest bar
plt.ylim(0, max_y * 1.1)  # Add 10% extra space above the tallest bar

# Adjust layout for clarity
plt.tight_layout()

# Display the plot
plt.show()



# Plot revenue

# Define a function to format the y-axis labels
def format_yaxis(value, tick_number):
    if value >= 1_000_000:  # Format as millions
        if value % 1_000_000 == 0:  # Check if it's an exact multiple of 1M
            return f'{int(value / 1_000_000)}M'  # Display as a whole number (e.g., 1M, 2M)
        else:
            return f'{value / 1_000_000:.2f}M'  # Display fractional millions (e.g., 1.25M)
    elif value >= 1_000:  # Format as thousands
        return f'{int(value / 1_000)}K'
    else:  # Keep as is for smaller values
        return f'{int(value)}'

revenue = ['first_touch_revenue', 'last_touch_revenue', 'linear_touch_revenue', 'markov_revenue']
revenue_labels = ['First Touch', 'Last Touch', 'Linear Touch', 'Markov Chain']  # Custom labels for the legend
ax = att_model_comb[revenue].plot(
    kind='bar', 
    figsize=(10, 9), 
    width=0.8, 
    color=colors)

# Customize x-axis labels and rotation
plt.xticks(range(len(att_model_comb)), att_model_comb['channel'], rotation=45)

# Add titles and axis labels
plt.title('Revenue per Channel')
plt.ylabel('Revenue in Million')
plt.xlabel('Channel')

# Add a legend with custom labels
plt.legend(revenue_labels, title="Attribution Type")


# Automatically add data labels, transformed to 'K'
for container in ax.containers:
    ax.bar_label(
        container,
        labels=[f'{int(label / 1000)}K' for label in container.datavalues],  # Divides by 1000 the revenue value and appends a 'K' to each label
        label_type='edge',
        padding=2,
        fontsize=8
    )

# Adjust the y-axis range to add more space above the bars
max_y = max([bar.get_height() for container in ax.containers for bar in container])  # Find the tallest bar
plt.ylim(0, max_y * 1.1)  # Add 20% extra space above the tallest bar

# Apply the custom y-axis formatter
ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis))

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# Plotting Conversion rate per Channel

# %%
# Define a function to format the y-axis labels
def format_yaxis(value, tick_number):
    return f'{value * 100:.0f}%'  # Multiply by 100 and format as an integer with a '%' symbol

# Extract the columns with _rate to plot
rate_columns = ['first_touch_conversions_rate','last_touch_conversions_rate','linear_touch_conversions_rate','markov_conversions_rate']
rate_labels = ['First Touch', 'Last Touch', 'Linear Touch', 'Markov Chain']  # Custom labels for the legend

# Create the grouped bar plot
ax = conversion_rates[rate_columns].plot(
    kind='bar',
    figsize=(10, 8),
    width=0.8,
    color=colors[:len(rate_columns)]  # Ensure proper number of colors
)

# Set the x-tick labels based on 'channel'
plt.xticks(
    ticks=range(len(conversion_rates)),  # Positions for x-ticks
    labels=conversion_rates['channel'],  # Channel names from DataFrame
    rotation=45  # Rotate channel names for readability
)

# Add a title, x-label, and y-label
plt.title('Conversion Rates per Channel')
plt.xlabel('Channel', fontsize=12)
plt.ylabel('Conversion Rate', fontsize=12)

# Add a legend for the rate columns
plt.legend(rate_labels, title="Attribution Type", fontsize=10)

# Automatically add data labels to the bars
for container in ax.containers:
    ax.bar_label(container,
                 labels=[f'{value * 100:.2f}%' for value in container.datavalues],  # Multiply values by 100 and add '%' symbol
                 label_type='edge',
                 padding=2,
                 fontsize=7)

# Adjust the y-axis range to add more space above the bars
max_y = max([bar.get_height() for container in ax.containers for bar in container])  # Find the maximum bar height
plt.ylim(0, max_y * 1.1)  # Set the y-axis range to 10% higher than the tallest bar

# Apply the custom y-axis formatter
ax.yaxis.set_major_formatter(FuncFormatter(format_yaxis))

# Adjust layout for clarity
plt.tight_layout()

# Display the plot
plt.show()


