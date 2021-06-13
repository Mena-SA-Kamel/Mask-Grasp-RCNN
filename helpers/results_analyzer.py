import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.ticker import MaxNLocator
# plt.style.use(['science','ieee'])

class ResultAnalyzer:
    def __init__(self, data_dir, experiment_name):
        self.data_dir = data_dir
        self.experiment_name = experiment_name
        self.total_num_objects = 0
        self.num_baskets = 0
        self.num_passes_exp_1 = 0
        self.num_passes_exp_0 = 0
        self.data_columns = ["date", "num_basket_objects", "object_iterator", "object_name", "Y1", "X1", "Y2", "X2",
                             "gaze_x", "gaze_y", "t_start", "t_select", "t_close", "t_completion", "NOSC", "NGSC",
                             "NAC", "TTA", "TTG", "OST", "TTT", "success", "prosthetic_err", "obj_select_err",
                             "grasp_config_err", "aperture_err"]
        self.flattened_data = []
        self.data_loaded = False
        self.data_frame = None
        self.exp_0_TAR = 0
        self.exp_1_TAR = 0
        self.exp_0_errors = None
        self.exp_1_errors = None
        self.exp_0_timing = None
        self.exp_1_timing = None
        self.exp_1_minus_0_timing = None
        self.basket_sizes = None

    def parse_log_dir(self):
        # Only parse the log directory if the data has not been loaded into self.flattened_data (Checking for
        # self.data_loaded flag)
        image_center_crop_size = 384
        self.basket_sizes = []
        if not self.data_loaded:
            trial_folders = os.listdir(self.data_dir)
            for folder in trial_folders:
                if "Miracast" in folder or ".DS_Store" in folder:
                    continue
                if self.experiment_name in folder:
                    folder_path = os.path.join(log_dir, folder)
                    json_file_name = folder + '.json'
                    json_path = os.path.join(folder_path, json_file_name)
                    data = self.load_from_json(json_path)
                    self.num_baskets += 1
                    basket_objects = data['objects']
                    num_basket_objects = data['num_objects']
                    experiment_date = data['date']
                    self.basket_sizes.append(num_basket_objects)
                    for object_iterator, object in enumerate(basket_objects):
                        data[object]['gaze_y'] += image_center_crop_size
                        failures = data[object]['fail_reason']
                        fail_reasons = np.array([failures['prosthetic_err'],
                                                 failures['obj_select_err'],
                                                 failures['grasp_config_err'],
                                                 failures['aperture_err']])
                        data_row = [experiment_date, num_basket_objects, object_iterator, object]
                        for key in data[object].keys():
                            if key == "fail_reason":
                                data[object][key] = fail_reasons.tolist()
                            if type(data[object][key]) is list:
                                for val in data[object][key]:
                                    data_row.append(val)
                                continue
                            data_row.append(data[object][key])
                        # data_row = np.array(data_row).reshape(1, -1)
                        self.flattened_data.append(data_row)
            self.flattened_data = self.flattened_data[1:]
            self.data_loaded = True

    def create_pandas_df(self):
        self.data_frame = pd.DataFrame(data=self.flattened_data, columns=self.data_columns)

    def load_from_json(self, file_path):
        with open(file_path, "r") as read_file:
            return json.load(read_file)

    def compute_TARs(self):
        self.num_passes_exp_0 = len(self.data_frame[self.data_frame['exp_0_verdict']])
        self.exp_0_TAR = self.num_passes_exp_0 / self.total_num_objects
        self.num_passes_exp_1 = len(self.data_frame[self.data_frame['exp_1_verdict']])
        self.exp_1_TAR = self.num_passes_exp_1 / self.total_num_objects

    def generate_verdicts(self):
        self.total_num_objects = len(self.data_frame)
        df = results.data_frame
        # exp_0_verdicts = ~df['prosthetic_err'] & ~df['obj_select_err'] & ~df['grasp_config_err'] & ~df['aperture_err'] \
        #                  & df['success'] & (df['NOSC'] == 0) & (df['NAC'] == 0)
        exp_0_verdicts = df['success'] & (df['NOSC'] == 0) & (df['NAC'] == 0)
        self.data_frame = df.assign(exp_0_verdict=exp_0_verdicts)
        exp_1_verdicts = self.data_frame['success']
        self.data_frame = self.data_frame.assign(exp_1_verdict=exp_1_verdicts)

    def analyze_error_sources(self):
        simplified_df = self.data_frame[['NOSC', 'NAC', 'prosthetic_err', 'obj_select_err', 'grasp_config_err',
                                         'aperture_err', 'exp_0_verdict', 'exp_1_verdict']]
        self.exp_0_errors = simplified_df.groupby('exp_0_verdict').sum().drop(['exp_1_verdict'], axis=1)
        self.exp_1_errors = simplified_df.groupby('exp_1_verdict').sum().drop(['exp_0_verdict'], axis=1)

        # NAC was used whenever an object slipped or when the aperture was incorrect
        # NOSC was incremented when the user reselected an object due to wrong
        # object selection or grasp configuration
        sources_of_error = simplified_df.to_numpy()
        error_combos = []
        combination_counts = {}
        count_anomaly_t1 = 0
        count_anomaly_t2 = 0

        for row in sources_of_error:
            row_errors = ""
            NOSC, NAC, PE, OSE, GCE, AE, exp_0_verdict, exp_1_verdict = row
            if PE:
                row_errors += "PE_"
            if OSE:
                row_errors += "OSE_"
            if GCE:
                row_errors += "GCE_"
            if AE:
                row_errors += "AE_"
            if not (row_errors == ""):
                try:
                    combination_counts[row_errors] += 1
                except:
                    combination_counts[row_errors] = 1
            error_combos.append(row_errors)
            # print(row, row_errors)
        print(simplified_df['NOSC'].sum())
        print(simplified_df['NAC'].sum())
        self.data_frame['error_combinations'] = error_combos

        # Plotting the total sources of error
        error_combination = self.data_frame['error_combinations'].value_counts()[1:]
        error_combination_labels = self.decode_categories(list(error_combination.index))

        bar_width = 0.7
        fig, ax = plt.subplots()
        exp_0 = ax.bar(error_combination_labels, error_combination, bar_width)
        ax.set_ylabel('Count',fontname="Times New Roman")
        ax.set_xlabel('Error Sources',fontname="Times New Roman")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.2)
        plt.show(block=False)


    def analyze_effect_of_object_type(self):
        simplified_df = self.data_frame[['object_name','NOSC', 'NAC', 'prosthetic_err', 'obj_select_err', 'grasp_config_err',
                                         'aperture_err', 'exp_0_verdict', 'exp_1_verdict']]
        simplified_df['object_name'] = simplified_df['object_name'].str.replace(' - UR', '')
        simplified_df['object_name'] = simplified_df['object_name'].str.replace(' - RDM', '')
        simplified_df['object_name'] = simplified_df['object_name'].str.title()
        object_class_occurence = simplified_df['object_name'].value_counts()
        simplified_df['correction_sum'] = simplified_df['NAC'] + simplified_df['NOSC']
        simplified_df.groupby(['exp_0_verdict', 'exp_1_verdict','object_name']).sum().sort_values('correction_sum')

        # fig = go.Figure()
        # fig.add_trace(go.Bar(
        #     x=object_class_occurence.index, y=object_class_occurence,
        # ))
        # fig.update_layout(font_family="Times New Roman",
        #                   xaxis_title="Object Class",
        #                   yaxis_title="Count",
        #                   font=dict(size=25),
        #                   template='plotly_white')
        # fig.show()

        bar_width = 0.7
        fig, ax = plt.subplots()
        ax.bar(object_class_occurence.index, object_class_occurence, bar_width)
        ax.set_ylabel('Count', fontname="Times New Roman")
        ax.set_xlabel('Object Class', fontname="Times New Roman")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.2)
        plt.xticks(rotation='vertical')
        plt.show()

    def analyze_effect_of_num_objects(self):
        simplified_df = self.data_frame[['num_basket_objects', 'object_iterator', 'exp_0_verdict', 'exp_1_verdict']]
        num_objects_remaining = simplified_df['num_basket_objects'] - simplified_df['object_iterator']
        simplified_df['num_objects_remaining'] = num_objects_remaining

        # need to find the distribution of object indices
        object_distrib = simplified_df['num_objects_remaining'].value_counts()
        simplified_df = simplified_df.drop(['num_basket_objects', 'object_iterator'], axis=1)
        num_passes = simplified_df.groupby('num_objects_remaining').sum()

        # need to divide the num_passes for each object index by the object distribution at that reset_index
        num_passes_np = num_passes.reset_index().to_numpy()
        results = np.zeros((num_passes_np.shape[0], 4))
        results[:,0] = num_passes_np[:,0]
        for i, row in enumerate(num_passes_np):
            objects_remaining, exp_0_verdict, exp_1_verdict = row
            results[i][1] = exp_0_verdict / object_distrib[objects_remaining]
            results[i][2] = exp_1_verdict / object_distrib[objects_remaining]
            results[i][3] = object_distrib[objects_remaining]

        results = results[:-1]
        results[:, 1:3]*=100
        df = pd.DataFrame(data=results, columns=['objects_in_scene', 'exp_0_TAR', 'exp_1_TAR', 'number_of_occurence'])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Full Autonomous Control',
            x=df['objects_in_scene'], y=df['exp_0_TAR'],
        ))
        fig.add_trace(go.Bar(
            name='Semi-Autonomous Control',
            x=df['objects_in_scene'], y=df['exp_1_TAR'],
        ))

        fig.update_layout(barmode='group',
                          font_family="Times New Roman",
                          xaxis_title="Number of Objects in the Scene",
                          yaxis_title="Task Accomplishment Rate (%)",
                          font=dict(size=25),
                          template='plotly_white')
        fig.show()

        index = np.arange(1,6)
        bar_width = 0.4

        fig, ax = plt.subplots()
        exp_0 = ax.bar(df['objects_in_scene'], df['exp_0_TAR'], bar_width,
                       label="Full Autonomous Control",)
        exp_1 = ax.bar(df['objects_in_scene'] + bar_width, df['exp_1_TAR'], bar_width,
                       label="Semi-Autonomous Control",
                       capsize=3)
        # exp_1_0 = ax.bar(index + 2*bar_width, timing_mean_exp_1_minus_0, bar_width, yerr=timing_std_exp_1_minus_0, label="Semi-Autonomous Control 2")

        ax.set_ylabel('Count', fontname="Times New Roman")
        ax.set_xlabel('Number of Objects in the Scene',fontname="Times New Roman")
        # ax.set_title('Comparing')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(df['objects_in_scene'])
        ax.legend(loc='upper right')
        ax.set_ylim([0, 120])
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.2)

        plt.show()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['objects_in_scene'], y=df['number_of_occurence'],
        ))
        fig.update_layout(font_family="Times New Roman",
                          xaxis_title="Number of Objects in the Scene",
                          yaxis_title="Count",
                          font=dict(size=25),
                          template='plotly_white')
        fig.show()

        bar_width = 0.7
        fig, ax = plt.subplots()
        ax.bar(df['objects_in_scene'], df['number_of_occurence'], bar_width)
        ax.set_ylabel('Count', fontname="Times New Roman")
        ax.set_xlabel('Number of Objects in the Scene', fontname="Times New Roman")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.2)
        # plt.xticks(rotation='vertical')
        plt.show()

    def analyze_time_info(self):
        # Need to flag the cases where the hand was never commanded to close (t_close = 0)
        df = self.data_frame.drop(['num_basket_objects', 'object_iterator', 'NOSC', 'NAC', 'object_name', 'success', 'prosthetic_err',
                                   'obj_select_err', 'grasp_config_err', 'aperture_err'], axis=1)
        df = df[(df['TTA'] > 0) & (df['TTG'] > 0) & (df['OST'] > 0) & (df['TTT'] > 0)]
        exp_0_df = df[df['exp_0_verdict']].drop(['t_start', 't_select', 't_close', 't_completion'], axis=1)
        # Only consider the data points that were considered a fail in Experiment 0
        exp_1_df = df[df['exp_1_verdict']].drop(['t_start', 't_select', 't_close', 't_completion'], axis=1)
        exp_1_minus_0_df = df[df['exp_1_verdict'] & ~df['exp_0_verdict']].drop(['t_start', 't_select', 't_close',
                                                                                't_completion'], axis=1)
        self.exp_0_timing = exp_0_df
        self.exp_1_timing = exp_1_df
        self.exp_1_minus_0_timing = exp_1_minus_0_df

    def decode_categories(self, categories, space=0):
        output = []
        for category in categories:
            label = ""
            if "PE" in category:
                label += ("PE" + space*" " + ", ")
            if "OSE" in category:
                 label += ("OSE" + space*" " + ", ")
            if "GCE" in category:
                label += ("GCE" + space*" " + ", ")
            if "AE" in category:
                label += ("AE" + space*" " + ", ")
            if label != "":
                label = label[:-2]
            output.append(label)
        return output

    def plot_TAR_figures(self):
        # Experiment 0
        df = self.data_frame
        exp_0_errors_fail = df[df['exp_0_verdict'] == False]['error_combinations'].value_counts()
        exp_0_errors_fail_labels = df[df['exp_0_verdict'] == False]['error_combinations'].value_counts().index
        exp_0_errors_fail_labels = self.decode_categories(exp_0_errors_fail_labels)

        exp_0_errors_pass = df[df['exp_0_verdict'] == True]['error_combinations'].value_counts()[1:]
        exp_0_errors_pass_labels = df[df['exp_0_verdict'] == True]['error_combinations'].value_counts().index[1:]
        exp_0_errors_pass_labels = self.decode_categories(exp_0_errors_pass_labels, space=1)

        labels = ["Full Autonomous Control", "Pass", "Fail"] + list(exp_0_errors_pass_labels) + \
                 list(exp_0_errors_fail_labels)
        parents = ["", "Full Autonomous Control", "Full Autonomous Control"] + ["Pass"] * len(
            exp_0_errors_pass_labels) + ["Fail"] * len(exp_0_errors_fail_labels)
        num_passes = self.num_passes_exp_0
        total_num_objects = self.total_num_objects
        num_fails = total_num_objects - num_passes
        values = [total_num_objects, num_passes, num_fails] + list(exp_0_errors_pass) + list(exp_0_errors_fail)

        sizes = [25]*len(labels)
        fig = go.Figure(go.Sunburst(
            branchvalues='total',
            textinfo= 'label+value',
            labels=labels,
            parents=parents,
            values=values,
            insidetextorientation='radial',
            textfont=dict(size=sizes),
            rotation=35
        ))
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        fig.show()


        exp_0_errors_fail_labels.remove('')
        exp_0_errors_fail.pop('')
        sizes = [25] * len(exp_0_errors_fail_labels)
        fig = go.Figure(data=[go.Pie(labels=exp_0_errors_fail_labels,
                                     values=exp_0_errors_fail,
                                     textinfo='label+percent+value',
                                    textfont=dict(size=sizes))])
        fig.show()

        sizes = [25] * len(exp_0_errors_pass_labels)
        fig = go.Figure(data=[go.Pie(labels=exp_0_errors_pass_labels,
                                     values=exp_0_errors_pass,
                                     textinfo='label+percent+value',
                                     textfont=dict(size=sizes))])
        fig.show()


        # Experiment 0 - User interventions
        corrections = list(self.exp_0_errors.to_numpy()[0][:2])
        exp_0_errors = list(self.exp_0_errors.to_numpy()[0][2:])
        labels = ["Full Autonomous Control", "Pass", "Fail", "NACs", "NOSCs"]
        parents = ["", "Full Autonomous Control", "Full Autonomous Control", "Fail", "Fail"]
        num_passes = self.num_passes_exp_0
        total_num_objects = self.total_num_objects
        num_fails = total_num_objects - num_passes
        values = [total_num_objects, num_passes, num_fails] + corrections
        sizes = [25]*len(labels)
        fig = go.Figure(go.Sunburst(
            branchvalues='total',
            textinfo= 'label+value',
            labels=labels,
            parents=parents,
            values=values,
            insidetextorientation='radial',
            textfont=dict(size=sizes)
        ))
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        fig.show()


        # Experiment 1
        df = self.data_frame
        exp_1_errors_fail = df[df['exp_1_verdict'] == False]['error_combinations'].value_counts()
        exp_1_errors_fail_labels = df[df['exp_1_verdict'] == False]['error_combinations'].value_counts().index
        exp_1_errors_fail_labels = self.decode_categories(exp_1_errors_fail_labels)

        exp_1_errors_pass = df[df['exp_1_verdict'] == True]['error_combinations'].value_counts()[1:]
        exp_1_errors_pass_labels = df[df['exp_1_verdict'] == True]['error_combinations'].value_counts().index[1:]
        exp_1_errors_pass_labels = self.decode_categories(exp_1_errors_pass_labels, space=1)

        labels = ["Semi-Autonomous Control", "Pass", "Fail"] + list(exp_1_errors_pass_labels) + list(exp_1_errors_fail_labels)
        parents = ["", "Semi-Autonomous Control", "Semi-Autonomous Control"] + ["Pass"]*len(exp_1_errors_pass_labels) + ["Fail"]*len(exp_1_errors_fail_labels)
        num_passes = self.num_passes_exp_1
        total_num_objects = self.total_num_objects
        num_fails = total_num_objects - num_passes
        values = [total_num_objects, num_passes, num_fails] + list(exp_1_errors_pass) + list(exp_1_errors_fail)


        sizes = [25]*len(labels)
        fig = go.Figure(go.Sunburst(
            branchvalues='total',
            textinfo= 'label',
            labels=labels,
            parents=parents,
            values=values,
            insidetextorientation='radial',
            textfont=dict(size=sizes),
            rotation=5
        ))
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        fig.show()

        sizes = [25] * len(exp_1_errors_fail_labels)
        fig = go.Figure(data=[go.Pie(labels=exp_1_errors_fail_labels,
                                     values=exp_1_errors_fail,
                                     textinfo='label+percent+value',
                                     textfont=dict(size=sizes))])
        fig.show()

        sizes = [25] * len(exp_1_errors_pass_labels)
        fig = go.Figure(data=[go.Pie(labels=exp_1_errors_pass_labels,
                                     values=exp_1_errors_pass,
                                     textinfo='label+percent+value',
                                     textfont=dict(size=sizes))])
        fig.show()
        # Experiment 1
        exp_1_errors_fail = list(self.exp_1_errors.to_numpy()[0][2:])
        fail_corrections = list(self.exp_1_errors.to_numpy()[0][:2])
        exp_1_errors_pass = list(self.exp_1_errors.to_numpy()[1][2:])
        pass_corrections = list(self.exp_1_errors.to_numpy()[1][:2])
        labels = ["Semi-Autonomous Control", "Pass", "Fail", "NACs", "NOSCs", "NACs ", "NOSCs "]
        parents = ["", "Semi-Autonomous Control", "Semi-Autonomous Control", "Fail", "Fail", "Pass", "Pass"]
        num_passes = self.num_passes_exp_1
        total_num_objects = self.total_num_objects
        num_fails = total_num_objects - num_passes
        values = [total_num_objects, num_passes, num_fails] + fail_corrections  + pass_corrections
        sizes = [15]*len(labels)
        fig = go.Figure(go.Sunburst(
            branchvalues='total',
            textinfo= 'label+value',
            labels=labels,
            parents=parents,
            values=values,
            # insidetextorientation='radial',
            textfont=dict(size=sizes)
        ))
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        fig.show()

        labels = ["NAC", "NOSC"]
        sizes = [25] * len(labels)
        fig = go.Figure(data=[go.Pie(labels=labels,
                                     values=fail_corrections,
                                     textinfo='label+percent+value',
                                     textfont=dict(size=sizes))])
        fig.show()

        sizes = [25] * len(labels)
        fig = go.Figure(data=[go.Pie(labels=labels,
                                     values=pass_corrections,
                                     textinfo='label+percent+value',
                                     textfont=dict(size=sizes))])
        fig.show()

    def plot_basket_size_distribution(self):
        basket_sizes = pd.DataFrame(self.basket_sizes)
        unique_sizes = np.unique(basket_sizes)
        basket_sizes_distribution = pd.value_counts(self.basket_sizes)
        distribution = []
        for i in unique_sizes:
            distribution.append(basket_sizes_distribution[i])

        # fig = go.Figure()
        # fig.add_trace(go.Bar(
        #     x=unique_sizes, y=distribution,
        # ))
        # fig.update_layout(font_family="Times New Roman",
        #                   xaxis_title="Basket Size",
        #                   yaxis_title="Count",
        #                   font=dict(size=25),
        #                   template='plotly_white')
        # fig.show()

        bar_width = 0.7
        fig, ax = plt.subplots()
        ax.bar(unique_sizes, distribution, bar_width)
        ax.set_ylabel('Count', fontname="Times New Roman")
        ax.set_xlabel('Basket Size', fontname="Times New Roman")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.2)
        # plt.xticks(rotation='vertical')
        plt.show()


    def plot_error_percentages(self):
        exp_1_failure_errors = [45.23809524,28.57142857,19.04761905, 7.142857143]
        exp_2_failure_errors = [42.85714286,28.57142857,14.28571429,14.28571429]
        exp_1_pass_errors = [22.22222222,33.33333333,33.33333333,11.11111111]
        exp_2_pass_errors = [40.54054054,29.72972973,24.32432432,5.405405405]

        exp_1_failure_count = [19,12,8,3]
        exp_2_failure_count = [6,4,2,2]
        exp_1_pass_count = [2,3,3,1]
        exp_2_pass_count = [15,11,9,2]

        labels = ["PE", "GCE","OSE","AE"]
        fig = go.Figure()
        bar_width = [0.3]
        width = len(labels)*bar_width
        fig.add_trace(go.Bar(
            name='Exp. I - Errors that did not affect result',
            x=labels, y=exp_2_failure_errors,
        ))
        fig.add_trace(go.Bar(
            name='Exp. I - Errors that lead to failure',
            x=labels, y=exp_1_failure_errors,
        ))

        fig.update_layout(barmode='group',
                          font_family="Times New Roman",
                          xaxis_title="Error Sources",
                          yaxis_title="Percentage Relative to Sum of Errors (%)",
                          font=dict(size=25),
                          template='plotly_white'
                          )
        fig.show()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Exp. II - Errors that did not affect result',
            x=labels, y=exp_2_pass_errors,
        ))
        fig.add_trace(go.Bar(
            name='Exp. II - Errors that lead to failure',
            x=labels, y=exp_1_pass_errors,
        ))

        fig.update_layout(barmode='group',
                          font_family="Times New Roman",
                          xaxis_title="Error Sources",
                          yaxis_title="Percentage Relative to Sum of Errors(%)",
                          font=dict(size=25),
                          template='plotly_white'
                          )
        fig.show()
        #
        # fig = go.Figure()
        # bar_width = [0.3]
        # width = len(labels)*bar_width
        # fig.add_trace(go.Bar(
        #     name='Exp. I - Errors that lead to failure',
        #     x=labels, y=exp_1_failure_count,
        # ))
        # fig.add_trace(go.Bar(
        #     name='Exp. I - Errors that did not affect result',
        #     x=labels, y=exp_2_failure_count,
        # ))
        # fig.update_layout(barmode='group',
        #                   font_family="Times New Roman",
        #                   xaxis_title="Error Sources",
        #                   yaxis_title="Number of Occurence",
        #                   font=dict(size=25),
        #                   template='plotly_white'
        #                   )
        # fig.show()
        #
        # fig = go.Figure()
        # fig.add_trace(go.Bar(
        #     name='Exp. I - Errors that did not affect result',
        #     x=labels, y=exp_1_pass_count,
        # ))
        # fig.add_trace(go.Bar(
        #     name='Exp. II - Errors that did not affect result',
        #     x=labels, y=exp_2_pass_count,
        # ))
        # fig.update_layout(barmode='group',
        #                   font_family="Times New Roman",
        #                   xaxis_title="Error Sources",
        #                   yaxis_title="Number of Occurence",
        #                   font=dict(size=25),
        #                   template='plotly_white'
        #                   )
        # fig.show()

    def plot_timing_figures(self):
        labels = list(self.exp_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1).mean().index)
        timing_mean_exp_0 = list(self.exp_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1).mean() / 1000.0)
        timing_mean_exp_1 = list(self.exp_1_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1).mean() / 1000.0)
        timing_std_exp_0 = list(self.exp_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1).std() / 1000.0)
        timing_std_exp_1 = list(self.exp_1_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1).std() / 1000.0)

        fig = go.Figure()
        bar_width = [0.3]
        width = len(labels)*bar_width
        fig.add_trace(go.Bar(
            name='Full Autonomous Control',
            x=labels, y=timing_mean_exp_0,
            error_y=dict(type='data', array=timing_std_exp_0),
        ))
        fig.add_trace(go.Bar(
            name='Semi-Autonomous Control',
            x=labels, y=timing_mean_exp_1,
            error_y=dict(type='data', array=timing_std_exp_1),
        ))
        fig.update_layout(barmode='group',
                          font_family="Times New Roman",
                          xaxis_title="Timing Metrics",
                          yaxis_title="Time (s)",
                          font=dict(size=25),
                          template='plotly_white'
                          )
        fig.show()


        timing_mean_exp_1_minus_0 = list(self.exp_1_minus_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1)
                                         .mean() / 1000.0)
        timing_std_exp_1_minus_0 = list(self.exp_1_minus_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1)
                                         .std() / 1000.0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Full Autonomous Control',
            x=labels, y=timing_mean_exp_0,
            error_y=dict(type='data', array=timing_std_exp_0),
        ))
        fig.add_trace(go.Bar(
            name='Semi-Autonomous Control',
            x=labels, y=timing_mean_exp_1,
            error_y=dict(type='data', array=timing_std_exp_1),
        ))
        fig.add_trace(go.Bar(
            name='Semi-Autonomous Control - Worst Case Scenario',
            x=labels, y=timing_mean_exp_1_minus_0,
            error_y=dict(type='data', array=timing_std_exp_1_minus_0),
        ))
        fig.update_layout(barmode='group',
                          font_family="Times New Roman",
                          xaxis_title="Timing Metrics",
                          yaxis_title="Time (s)",
                          font=dict(size=25),
                          template='plotly_white')
        fig.show()


        # index = np.arange(4)
        # bar_width = 0.2
        #
        # fig, ax = plt.subplots()
        # exp_0 = ax.bar(index, timing_mean_exp_0, bar_width,
        #                yerr=timing_std_exp_0,
        #                label="Full Autonomous Control",
        #                capsize=3)
        # exp_1 = ax.bar(index + bar_width, timing_mean_exp_1, bar_width,
        #                yerr=timing_std_exp_1,
        #                label="Semi-Autonomous Control",
        #                capsize=3)
        # # exp_1_0 = ax.bar(index + 2*bar_width, timing_mean_exp_1_minus_0, bar_width, yerr=timing_std_exp_1_minus_0, label="Semi-Autonomous Control 2")
        #
        # # ax.set_xlabel('')
        # ax.set_ylabel('Time (s)',fontname="Arial")
        # # ax.set_title('Comparing')
        # ax.set_xticks(index + bar_width / 2)
        # ax.set_xticklabels(labels)
        # ax.legend(loc='upper left')
        #
        # plt.show()
        # import code;
        # code.interact(local=dict(globals(), **locals()))



log_dir = "Experiment Logs"
experiment_name = "Experiment_1"
results = ResultAnalyzer(log_dir, experiment_name)
results.parse_log_dir()
results.create_pandas_df()
df = results.data_frame

# Task Accomplishment Rates
# Need to omit entries that do not have a start time
df_filtered = df[(df['t_start'] > 0) & (df['TTA'] > 0) & (df['TTG'] > 0) & (df['OST'] > 0) & (df['TTT'] > 0)]

# Selecting relevant rows:
df_relevant_fields = df_filtered[['num_basket_objects', 'object_iterator', 'object_name', 't_start', 't_select', 't_close',
       't_completion', 'NOSC', 'NAC', 'TTA', 'TTG', 'OST', 'TTT',
       'success', 'prosthetic_err', 'obj_select_err', 'grasp_config_err',
       'aperture_err']]

results.data_frame = df_relevant_fields
results.generate_verdicts()
results.compute_TARs()
results.analyze_error_sources()
results.analyze_time_info()
# results.analyze_effect_of_object_type()
results.analyze_effect_of_num_objects()
# results.plot_basket_size_distribution()
# results.plot_timing_figures()
# results.plot_error_percentages()
# results.plot_TAR_figures()


print("\n-----------------------RESULTS SUMMARY----------------------\n")
print("Total Number of Objects: " + str(results.total_num_objects))
print("Total Number of Baskers: " + str(results.num_baskets))

print("\n---------------Experiment 0 - No User Control---------------")
print("Total Number of Passes for Experiment 0 (No user input): " + str(results.num_passes_exp_0))
print("Task Accomplishment Rate for Experiment 0 (No user input): " +  str(round(results.exp_0_TAR * 100, 2)) + "%")
# print("Timing for Experiment 0: " + str(TTT_mean_exp_0) + "+-" + str(TTT_std_exp_0) + " s")
print("Failure Distribution: \n" + str(results.exp_0_errors))
timing_info_exp_0 = results.exp_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1)
print("Mean Timing for Experiment 0: \n" + str(timing_info_exp_0.mean()/1000.0))
print("Standard Deviation of Timing for Experiment 0: \n" + str(timing_info_exp_0.std()/1000.0))
print("------------------------------------------------------------\n")

print("\n---------------Experiment 1 - Proposed System---------------")
print("Total Number of Passes for Experiment 1: " + str(results.num_passes_exp_1))
print("Task Accomplishment Rate for Experiment 1: " + str(round(results.exp_1_TAR * 100, 2)) + "%")
print("Failure Distribution: \n" + str(results.exp_1_errors))
timing_info_exp_1 = results.exp_1_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1)
print("Mean Timing for Experiment 1: \n" + str(timing_info_exp_1.mean()/1000.0))
print("Standard Deviation of Timing for Experiment 1: \n" + str(timing_info_exp_1.std()/1000.0))

print("\n------Timing on objects that failed in experiment 0---------")

timing_info_exp_1_minus_0 = results.exp_1_minus_0_timing.drop(['exp_0_verdict', 'exp_1_verdict'], axis=1)
print("Number of objects that failed in experiment 0, but passed in experiment 1: " + str(len(timing_info_exp_1_minus_0)))
print("Mean Timing for Experiment 1 minus 0: \n" + str(timing_info_exp_1_minus_0.mean()/1000.0))
print("Standard Deviation of Timing for Experiment 1 minus 0: \n" + str(timing_info_exp_1_minus_0.std()/1000.0))

print("------------------------------------------------------------\n")

# import code; code.interact(local=dict(globals(), **locals()))



# output = pd.read_pickle("flattened_data.pkl")
#
# # PLOTTING
# labels = ["Task Accomplishment Rate", "Task Failure Rate"]
# exp_0_tar = round(results.exp_0_TAR * 100, 2)
# exp_0_tar_values = [exp_0_tar, 100 - exp_0_tar]
# exp_1_tar = round(results.exp_1_TAR * 100, 2)
# exp_1_tar_values = [exp_1_tar, 100 - exp_1_tar]
#
# # Create subplots: use 'domain' type for Pie subplot
# fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
# fig.add_trace(go.Pie(labels=labels, values=exp_0_tar_values, name="Experiment I", textfont=dict(size=[25, 25])),
#               1, 1)
# fig.add_trace(go.Pie(labels=labels, values=exp_1_tar_values, name="Experiment II", textfont=dict(size=[25, 25])),
#               1, 2)
#
# # Use `hole` to create a donut-like pie chart
# fig.update_traces(hole=.4, hoverinfo="label+percent+name")
#
# fig.update_layout(
#     font_family="Times New Roman",
#     annotations=[dict(text='Experiment I', x=0.18, y=0.5, font_size=25, showarrow=False),
#                  dict(text='Experiment II', x=0.82, y=0.5, font_size=25, showarrow=False)],
#     legend = dict(font = dict(family = "Times New Roman", size = 25, color = "black")))
# fig.show()





#
# # Grouping based on success and sources of error to determine Task Accomplishment rates
# grouping = df_filtered.groupby(['success', 'prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err']).mean()
# error_distribution = df_filtered[['prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err', 'success']].groupby(['success']).mean().reset_index()
# df_filtered[['prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err', 'success']].sum().reset_index()
# df_filtered[['prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err', 'success']].groupby(['success']).sum().reset_index()
#
# # Need data frames for experiments 1 and 0, with a redefined success column
# # Experiment 0: A successful entry is one where there are no failures specified, and the grasping task actually happened
# # IE: ['prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err'] are all false, and success is True
#
#
# df_experiment_0 = df_filtered
# df_experiment_0.loc[~df_filtered['prosthetic_err']
#             & ~df_filtered['obj_select_err']
#             & ~df_filtered['grasp_config_err']
#             & ~df_filtered['aperture_err']]['success'] = False
#
#
# # Plotting number of times objects were grasped
# df['object_name'] = df['object_name'].str.replace("- UR", "").str.replace("- RDM", "")
# df['object_name'].value_counts().plot.bar(x='Object Category', y='Number of Occurences', rot=90)
#
# # Plotting the task accomplishment times for each object
# df['TTT'] /= 1000
# df[['object_name', 'TTT']].groupby(df['object_name']).mean()
#

# # Grouping
# pd.value_counts(df['success']).plot.bar()
# df['success'].mean()
# df_simplified = df.drop(df.columns[:10].tolist(), axis=1)
# df_simplified.groupby(['success', 'prosthetic_err', 'obj_select_err', 'grasp_config_err', 'aperture_err']).mean()
#
# import code; code.interact(local=dict(globals(), **locals()))
