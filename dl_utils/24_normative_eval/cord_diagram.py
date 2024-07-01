from bokeh.sampledata.les_mis import data
import holoviews as hv
from holoviews import opts, dim
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import json
import seaborn as sns


# Should the diagram be plotted with 'bokeh' or 'matplotlib'?
hv.extension('bokeh')
# How large should be the diagram?
hv.output(size=300)

# Data set
nodes = pd.DataFrame(data['nodes'])
nodes_gt = hv.Dataset(pd.DataFrame(data['nodes']), 'index')

source_0 = ['RQI', 'AHI', 'CACI', 'H(RQI,CACI)','RQI,AHI,CACI']
destination_0 = ['Absent', 'Craniatomy', 'Dural', 'Edema', 'Encephalomalacia', 'Enlarged','Intraventricular', 'Lesions', 'Posttreatment', 'Resection', 'Sinus', 'WML', 'Mass', 'Stroke (S)', 'Stroke (M)', 'Stroke (L)']
data = {
    # 'RQI': [0.79, 0.39, 0.47, 0.36, 0.45, 0.77, 0.87, 0.78, 0.45, 0.43, 0.48, 0.77, 1.02, 1.16],
    # 'RQI': [0.88, 0.04, 0.17, 0.6, 0.64, 0.66, 0.75, 0.85, 0.14, 0.02, 0.56, 0.89, 0.89, 0.9],
    # 'AHI': [0.02, 0.03, 0.07, 0.04, 0.06, 0.27, 0.15, 0.14, 0.09, 0.06, 0.03, 0.16, 0.43, 0.43],
    # 'CACI': [0.06, 0.22, 0.26, 0.15, 0.19, 0.22, 0.21, 0.31, 0.23, 0.20, 0.30, 0.28, 0.49, 0.45],
    'RQI' :[0.88, 0.90, 0.14, 0.75, 0.85, 0.60, 0.17, 0.02, 0.56, 0.89, 0.89, 0.66, 0.64, 0.04],
    'AHI': [0.00, 0.49, 0.00, 0.10, 0.03, 0.00, 0.00, 0.00, 0.00, 0.06, 0.44, 0.16, 0.00, 0.00],
    'CACI': [0.06, 0.45, 0.23, 0.21, 0.31, 0.15, 0.26, 0.20, 0.30, 0.28, 0.49, 0.22, 0.19, 0.22],

    #'FastMRI': [2.06, 10.21, 16.10, 9.55, 10.01, 40.78, 15.44, 23.88, 10.38, 10.46, 18.34, 30.37, 40.28, 41.77],
    'Absent': [0.00, 0.25, 0.00, 0.00, 0.14, 0.00, 0.00, 0.00, 0.00, 0.17, 0.18, 0.15, 0.00, 0.00],
    'Craniatomy': [0.02, 0.32, 0.16, 0.16, 0.08, 0.18, 0.16, 0.15, 0.15, 0.16, 0.34, 0.30, 0.11, 0.12],
    'Dural': [0.01, 0.43, 0.13, 0.47, 0.33, 0.04, 0.21, 0.04, 0.32, 0.51, 0.51, 0.45, 0.11, 0.12],
    'Edema': [0.00, 0.41, 0.01, 0.03, 0.28, 0.00, 0.03, 0.01, 0.09, 0.47, 0.38, 0.30, 0.00, 0.00],
    'Encephalomalacia': [0.00, 0.67, 0.00, 0.29, 0.40, 0.00, 0.22, 0.00, 0.40, 0.67, 1.00, 0.67, 0.00, 0.00],
    'Enlarged': [0.00, 0.39, 0.01, 0.14, 0.46, 0.16, 0.38, 0.06, 0.08, 0.19, 0.81, 0.73, 0.07, 0.04],
    'Intraventricular': [0.00, 0.67, 0.20, 0.13, 0.29, 0.12, 0.15, 0.15, 0.29, 0.33, 0.40, 0.50, 0.18, 0.15],
    'Lesions': [0.00, 0.35, 0.02, 0.03, 0.12, 0.01, 0.03, 0.01, 0.07, 0.18, 0.26, 0.23, 0.01, 0.02],
    'Posttreatment': [0.01, 0.35, 0.11, 0.11, 0.12, 0.08, 0.08, 0.10, 0.12, 0.17, 0.27, 0.24, 0.06, 0.10],
    'Resection': [0.00, 0.49, 0.10, 0.14, 0.33, 0.14, 0.13, 0.17, 0.23, 0.38, 0.49, 0.54, 0.20, 0.13],
    'Sinus': [0.09, 0.07, 0.16, 0.11, 0.01, 0.09, 0.14, 0.15, 0.12, 0.05, 0.05, 0.27, 0.11, 0.17],
    'WML': [0.00, 0.22, 0.00, 0.01, 0.05, 0.01, 0.00, 0.00, 0.01, 0.18, 0.02, 0.16, 0.02, 0.00],
    'Mass': [0.00, 0.50, 0.10, 0.08, 0.11, 0.08, 0.12, 0.07, 0.15, 0.28, 0.31, 0.25, 0.03, 0.08],
    'Stroke (S)': [0.0039, 0.0857, 0.0153, 0.0282, 0.035, 0.0127, 0.0071, 0.021, 0.0298, 0.0374, 0.0552, 0.0279, 0.0171,
                   0.0016],
    'Stroke (M)': [0.0382, 0.23378, 0.0856, 0.0958, 0.2397, 0.1038, 0.0678, 0.1861, 0.1521, 0.2395, 0.3039, 0.1993,
                   0.141, 0.0782],
    'Stroke (L)': [0.132, 0.3775, 0.23879, 0.2319, 0.5232, 0.2816, 0.1973, 0.3872, 0.36829, 0.4371, 0.5439, 0.4229,
                   0.3475, 0.205]
    # 'Stroke (S)': [0.39, 8.57, 1.53, 2.82, 3.50, 1.27, 0.71, 2.10, 2.98, 3.74, 5.52, 2.79, 1.71, 0.16],
    # 'Stroke (M)': [3.82, 23.38, 8.56, 9.58, 23.97, 10.38, 6.78, 18.61, 15.21, 23.95, 30.39, 19.93, 14.10, 7.82],
    # 'Stroke (L)':[13.20, 37.75, 23.88, 23.20, 52.32, 28.16, 19.73, 38.72, 36.83, 43.71, 54.39, 42.29, 34.75, 20.50]
    # 'Absent': [0.00, 0.00, 0.00, 0.00, 0.00, 15.38, 0.00, 14.29, 0.00, 0.00, 0.00, 16.67, 18.18, 25.00],
    # 'Craniatomy': [6.26, 14.66, 18.75, 19.19, 14.47, 34.78, 16.86, 14.04, 16.99, 18.48, 17.10, 20.39, 36.15, 37.03],
    # 'Dural': [4.76, 20.83, 29.75, 9.54, 19.31, 52.65, 47.02, 38.48, 22.70, 8.81, 37.85, 50.87, 51.31, 50.27],
    # # 'Dice': [3.43, 5.79, 5.58, 8.34, 10.84, 14.54, 8.13, 17.77, 7.21, 13.22, 12.16, 16.67, 21.39, 18.03],
    # 'Edema': [0.00, 4.07, 11.48, 3.44, 0.00, 45.56, 9.07, 35.51, 4.52, 4.63, 17.38, 49.46, 43.28, 45.78],
    # 'Encephalomalacia': [0.00, 0.00, 22.22, 0.00, 0.00, 66.67, 28.57, 40.00, 0.00, 0.00, 40.00, 66.67, 100.00, 66.67],
    # 'Enlarged': [0.00, 11.81, 44.75, 22.82, 15.70, 77.54, 22.70, 51.23, 5.91, 17.37, 15.77, 27.56, 80.51, 40.84],
    # 'Intraventricular': [0.00, 15.38, 15.38, 12.50, 18.18, 50.00, 13.33, 28.57, 20.00, 15.38, 28.57, 33.33, 40.00, 66.67],
    # 'Lesions': [2.27, 4.90, 6.07, 3.68, 3.97, 29.50, 5.32, 16.92, 5.05, 3.71, 10.94, 21.22, 27.31, 36.30],
    # 'Posttreatment': [3.81, 14.96, 11.76, 12.08, 9.44, 30.78, 14.32, 15.94, 14.76, 14.55, 15.51, 18.67, 29.65, 38.97],
    # 'Resection': [0.95, 16.13, 16.87, 17.78, 24.55, 54.32, 17.33, 33.21, 15.00, 24.23, 23.47, 37.86, 49.33, 49.44],
    # 'Sinus': [8.76, 16.67, 14.22, 9.16, 11.42, 26.67, 11.26, 1.72, 15.88, 15.11, 11.54, 5.35, 10.00, 14.29],
    # 'WML': [0.00, 0.00, 1.02, 1.58, 6.03, 15.50, 2.87, 8.14, 0.00, 0.00, 3.02, 17.70, 4.07, 22.16],
    # 'Mass': [0.00, 13.37, 16.97, 12.36, 7.08, 30.78, 12.01, 12.42, 14.12, 13.66, 17.24, 29.11, 33.89, 49.57],
    # 'Stroke (S)': [0.39, 0.16, 0.71, 1.27, 1.71, 2.79, 2.82, 3.50, 1.53, 2.10, 2.98, 3.74, 5.52, 8.57],
    # 'Stroke (M)': [3.82, 7.82, 6.78, 10.38, 14.10, 19.93, 9.58, 23.97, 8.56, 18.61, 15.21, 23.95, 30.39, 23.38],
    # 'Stroke (L)': [13.20, 20.50, 19.73, 28.16, 34.75, 42.29, 23.20, 52.32, 23.88, 38.72, 36.83, 43.71, 54.39, 37.75],
}

names = ['RQI', 'AHI', 'CACI', 'H(RQI,CACI)', 'RQI,AHI,CACI', 'Absent', 'Craniatomy', 'Dural', 'Edema', 'Encephalomalacia', 'Enlarged','Intraventricular', 'Lesions', 'Posttreatment', 'Resection', 'Sinus', 'WML', 'Mass', 'Stroke (S)', 'Stroke (M)', 'Stroke (L)']
groups = [0, 0, 0, 0,0, 1, 1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
nodes = pd.DataFrame({'name': np.asarray(names), 'group': np.asarray(groups)})
# # Create a DataFrame
df = pd.DataFrame(data)

# Calculate the harmonic mean between RQI, AHI, CACI for each method
df['H(RQI,CACI)'] = df.apply(lambda row:
                      scipy.stats.hmean([row['RQI'], row['CACI']]),
                      axis=1)
df['RQI,AHI,CACI'] = df.apply(lambda row:
                      scipy.stats.pmean([row['H(RQI,CACI)'], row['AHI']],1),
                      axis=1)
# df['HarMean2'] = df.apply(lambda row:
#                       scipy.stats.hmean([row['RQI'], row['CACI']]),
#                       axis=1)

df = df.sort_values(by=['RQI,AHI,CACI'],ascending=False)


# Initialize a MinMaxScaler
# scaler = MinMaxScaler()

# Select the columns to be normalized
cols_to_normalize = names

# Perform the normalization
# df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
#
# # Plot the correlation heatmap
# plt.figure(figsize=(10, 8))
first_3 = df[cols_to_normalize].head(5)
last_3 = df[cols_to_normalize].tail(0)
to_corr = pd.concat([last_3, first_3])
# to_corr = df[cols_to_normalize].iloc[5:9]
# correlation_matrix = df[cols_to_normalize].loc[0:12].corr(method='pearson')

correlation_matrix = to_corr.corr(method='pearson')
# correlation_matrix = df[cols_to_normalize].loc[0:12].corr(method='pearson')


sources = []
targets = []
colors = []
import numpy as np

leng_s = len(source_0)
pearson_corr = dict()
all_values = []
for source_id, source in enumerate(source_0):
    corr_source = correlation_matrix[source]
    pearson_corr[source] = dict()
    for dest_id, dest in enumerate(destination_0):
        value = corr_source[dest]
        sources.append(source_id)
        targets.append(dest_id + leng_s)
        if value < 0:
            value =0
        colors.append(value)
        pearson_corr[source][dest] = value
        all_values.append(value)
values = np.ones(len(colors))
# print(colors)
print(f'Min: {np.nanmin(all_values)}, Max: {np.nanmax(all_values)}')

# colors = ['1','2','3','1', '1','4', '10', '8']

links = pd.DataFrame({'source': sources, 'target': targets, 'value': values, 'color': colors})

# rq = correlation_matrix['RQI']
# rq4 = int(np.ceil(rq[4]*10))
#
# # Chord diagram

nodes_d = hv.Dataset(nodes, 'index')
chord = hv.Chord((links, nodes_d)).select(value=(1, None))
plasma10 = plt.get_cmap('copper', 20)
chord.opts(
    opts.Chord(cmap='Set3', edge_cmap=plasma10, edge_color=dim('color').str(), edge_line_width=(dim('color')*2).str(), bgcolor='black',label_text_color='white', colorbar=True,
               labels='name', node_color=dim('index').str()))
#edge_line_width=colors,
a= dim('value').str()
# Not needed in a jupyter notebook
# It shows the diagram when run in another IDE or from a python script:
from bokeh.plotting import show

show(hv.render(chord))
# with open('./corr_h32neg.txt', 'w') as file:
#     file.write(json.dumps(pearson_corr))  # use `json.loads` to do the revers
# to_corr.to_json('./corr_h32_input.txt')
    # file.write(json.dumps(to_corr))  # use `json.loads` to do the revers
# a= dim('sources')
sns.heatmap(correlation_matrix, annot=False, cmap='BrBG', vmin=-0.75, vmax=1)
# # plt.show()
plt.savefig('./heat_Avg.png', format='png', dpi=300, transparent=True)
plt.close()