import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.style
import matplotlib as mpl

mpl.style.use('dark_background')
with open('data.pkl', 'rb') as f:
    data = pkl.load(f)
with open('outcomes.pkl', 'rb') as f:
    outcomes_data = pkl.load(f)

original_data_count = len(data)

severity_metrics = ['Attitude', 'Paw Temperature', 'Vomiting', 'Gum Color', 'On Distemper Watch? (only mark on shift watch started)', 'Appetite', 'Feces', 'Drinking Water']
identifiers = ['Dog A#']
temporal_labels = ['Treatments Since Intake', 'Treatment Date', 'Treatment Shift']
outcomes = ['outcome']
time_series = data[identifiers + severity_metrics + temporal_labels + outcomes]
time_series['Treatment Shift'] = time_series['Treatment Shift'].replace(['AM', 'PM'], [0, 1])
#tmp = DatetimeIndex(['0000-00-00 12:00:00'], dtype='datetime64[ns]')
#print(tmp.head(1))
time_series['Treatment Date'] = time_series['Treatment Date'].add(time_series['Treatment Shift'].multiply(pd.to_timedelta(12, unit='h')))
time_series = time_series.drop('Treatment Shift', axis=1)

demo_metrics = ['Intake Weight (lbs)', 'Sex', 'Age at Intake (Weeks)']
demographics = outcomes_data[['Dog A#'] + demo_metrics].replace('Female', 0).replace('Male', 1).replace('Unknown', np.nan)

columns = np.array(time_series.columns)
columns[9] = 'stop'
columns[0] = 'id'
columns[11] = 'event'

time_series.columns = columns

# hack for time element
time_series['start'] = np.array(time_series['stop']) - 1

add = []
dates = []
shifts = []
for index, row in time_series.iterrows():
    add.append(np.array(demographics[demographics['Dog A#']==row['id']][demo_metrics])[0])
    
weights, genders, ages = np.transpose(add)

time_series['Intake Weight (lbs)'] = weights
time_series['Sex'] = genders
time_series['Age at Intake (Weeks)'] = ages

time_series = time_series.apply(pd.to_numeric)

for metric in time_series.columns:
    time_series = time_series[~np.isnan(time_series[metric])]

time_series_full = time_series.copy()
 
grp = time_series.groupby('id')
for name, group in grp:
    vals = np.array(group['event'])
    tmp = vals[0]
    vals = np.zeros(len(vals))
    vals[-1] = tmp
    time_series.loc[time_series['id'] == name, 'event'] = vals

#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
import warnings
warnings.filterwarnings('ignore')

from IPython.core import display as ICD

grp = time_series.groupby('id')
for name, group in grp:
    if np.array(group['event'])[-1] == 1.:
        ICD.display(group)
        break

eliminated_rows = original_data_count-len(time_series)
print('{0:.2f}% of (count={1}) rows eliminated due to nan values in demographics.'.format(
    (eliminated_rows)/original_data_count*100.,
    eliminated_rows))

from lifelines import CoxTimeVaryingFitter
%matplotlib inline
import matplotlib.style
import matplotlib as mpl

mpl.style.use('default')

# Using Cox Proportional Hazards model
ctvf = CoxTimeVaryingFitter()
ctvf.fit(time_series.drop(['Treatment Date'], axis=1), id_col="id", event_col="event", start_col="start", stop_col="stop", step_size=0.1, show_progress=True)
ctvf.print_summary()

ctvf.plot()