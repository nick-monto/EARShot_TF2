import tensorflow as tf,numpy as np,pandas as pd,_pickle as pickle
import os, io, gc, json
import argparse

from ProgressBar import progress

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

if not hp_Dict['Device'] is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp_Dict['Device']

class Analyzer:
    def __init__(
        self,
        path,
        absolute_Criterion= 0.7,
        relative_Criterion= 0.05,
        time_Dependency_Criterion= (10, 0.05),
        step_Cut= True
        ):
        self.result_Path = os.path.join(hp_Dict['Result_Path'], path).replace('\\', '/')

        self.absolute_Criterion = absolute_Criterion
        self.relative_Criterion = relative_Criterion
        self.time_Dependency_Criterion = time_Dependency_Criterion
        self.step_Cut = step_Cut

        self.Pattern_Metadata_Load()    # self.word_Index_Dict, self.step_Dict, self.max_Step, self.targets
        self.Category_Dict_Generate()   # self.category_Dict
        self.Adjusted_Length_Dict_Generate()    # self.adjusted_Length_Dict

        self.Analysis()

        self.parse_rt_file()
        self.parse_cf_file()


    def parse_rt_file(self):
        '''
        Reads the "RTs" file produced by EARShot and creates an Epoch-Accuracy
        dictionary for onset- and off-set relative, absolute, and time-dependent
        accuracies. Saves the result to a pickle.
        '''
        acc_file = os.path.join(self.result_Path, 'Test', 'RTs.txt').replace('\\', '/')
        data = pd.read_csv(acc_file,sep='\t')
        # dictionary to hold the results
        acc_data = {}.fromkeys(data.columns[10:].values)
        epochs = list(np.unique(data["Epoch"]))
        for meas in acc_data:
            acc_data[meas] = {}.fromkeys(np.unique(data["Epoch"]))

        for e in epochs:
            subframe = data[data["Epoch"] == e]
            for meas in acc_data:
                n_wrong = subframe[meas].isnull().sum()
                n_total = len(subframe[meas])
                acc_data[meas][e] = (n_total - n_wrong)/n_total

        # dump the results
        fname = os.path.join(self.result_Path, 'Test', 'ACC.pydb').replace('\\', '/')
        pickle.dump(acc_data,open(fname,'wb'))
        #col_order = ['Epoch']+list(acc_data.keys())
        #f = open(os.path.join(self.result_Path, 'Test', 'ACC.txt').replace('\\', '/'), 'w')
        # header
        #h_string = ','.join([k for k in col_order])+'\n'
        #f.write(h_string)
        # data
        #for e in epochs:
        #    w_s = ','.join([str(e)]+[str(acc_data[k][e]) for k in col_order[1:]])+'\n'
        #    f.write(w_s)
        #f.close()


    def parse_cf_file(self):
        '''
        Reads the "Category_Flows" file produced by EARShot and creates average
        similarities over (recurrent model) time to target, cohort, rhymes, and
        unrelated words in the lexicon, for each checkpointed epoch.

        I'm not handling speakers/word/etc. at all right now - this just
        averages over every correct target response in every epoch.

        Saves the resulting dictionary to a pickle.
        '''
        cf_file = os.path.join(self.result_Path, 'Test', 'Category_Flows.txt').replace('\\', '/')
        data = pd.read_csv(cf_file,sep='\t')

        # set up the dictionary to hold the results
        cat_data = {}.fromkeys(np.unique(data["Epoch"]))
        # there's nothing useful in the zero-epoch set
        cat_data.pop(0)
        for e in cat_data:
            cat_data[e] = {}.fromkeys(np.unique(data["Category"]))

        # now accumulate stats for all epochs > 0; ignore any incorrect responses
        for e in cat_data:
            epoch = data[(data["Epoch"] == e) & (data["Accuracy"] == True)]
            # appropriate selections
            for c in cat_data[e]:
                mean_series = epoch[epoch['Category'] == c].mean()
                cat_data[e][c] = mean_series.values[13:]

        # now dump the results
        fname = os.path.join(self.result_Path, 'Test', 'CS.pydb').replace('\\', '/')
        pickle.dump(cat_data,open('fname','wb'))
        #f = open(os.path.join(self.result_Path, 'Test', 'CS.txt').replace('\\', '/'), 'w')
        #for e in cat_data:
        #    for c in cat_data[e]:
        #        w_s = ','.join([str(e),c]+[str(x) for x in cat_data[e][c]])
        #        f.write(w_s+'\n')
        #f.close()


    def Analysis(self, batch_Steps= 200):
        result_File_List = sorted([ #Result files sorting
            os.path.join(self.result_Path, 'Test', x).replace('\\', '/')
            for x in os.listdir(os.path.join(self.result_Path, 'Test').replace('\\', '/'))
            if x.endswith('.pickle') and x != 'Metadata.pickle'
            ])

        reaction_Times = [
            '\t'.join(['{}'.format(x) for x in [
                'Epoch',
                'Word',
                'Identifier',
                'Pattern_Type',
                'Pronunciation',
                'Pronunciation_Length',
                'Uniqueness_Point',
                'Cohort_N',
                'Rhyme_N',
                'Neighborhood_N',
                'Onset_Absolute_RT',
                'Onset_Relative_RT',
                'Onset_Time_Dependent_RT',
                'Offset_Absolute_RT',
                'Offset_Relative_RT',
                'Offset_Time_Dependent_RT'
            ]])]
        category_Flows = [
            '\t'.join(['{}'.format(x) for x in [
                'Epoch',
                'Word',
                'Identifier',
                'Pattern_Type',
                'Pronunciation',
                'Pronunciation_Length',
                'Uniqueness_Point',
                'Cohort_N',
                'Rhyme_N',
                'Neighborhood_N',
                'Category',
                'Category_Count',
                'Accuracy'
            ] + list(range(self.max_Step))])]
        for result_File in result_File_List:
            with open(result_File, 'rb') as f:
                result_Dict = pickle.load(f)
            epoch = result_Dict['Epoch']
            infos = result_Dict['Info']
            outputs = result_Dict['Result'] #[Batch, Steps, Dims]

            for index, (output, (word, identifier, pattern_Type)) in enumerate(zip(outputs, infos)):
                data = self.Data_Generate(output, word, identifier, batch_Steps) #[Num_Words, Steps]
                rt_Dict = self.RT_Generate(word, identifier, data)
                category_Flow_Dict = self.Category_Flow_Generate(word, data)
                reaction_Times.append(
                    '\t'.join(['{}'.format(x) for x in [
                        epoch,
                        word,
                        identifier,
                        pattern_Type,
                        '.'.join(self.pattern_Metadata_Dict['Pronunciation_Dict'][word]),
                        len(self.pattern_Metadata_Dict['Pronunciation_Dict'][word]),
                        self.adjusted_Length_Dict[word],
                        len(self.category_Dict[word, 'Cohort']),
                        len(self.category_Dict[word, 'Rhyme']),
                        len(self.category_Dict[word, 'DAS_Neighborhood']),
                        rt_Dict['Onset', 'Absolute'],
                        rt_Dict['Onset', 'Relative'],
                        rt_Dict['Onset', 'Time_Dependent'],
                        rt_Dict['Offset', 'Absolute'],
                        rt_Dict['Offset', 'Relative'],
                        rt_Dict['Offset', 'Time_Dependent']
                        ]])
                    )

                for category in ["Target", "Cohort", "Rhyme", "Unrelated", "Other_Max"]:
                    if category == "Other_Max":
                        category_Count = np.nan
                    else:
                        category_Count = len(self.category_Dict[word, category])
                    category_Flows.append(
                        '\t'.join(['{}'.format(x) for x in [
                            epoch,
                            word,
                            identifier,
                            pattern_Type,
                            '.'.join(self.pattern_Metadata_Dict['Pronunciation_Dict'][word]),
                            len(self.pattern_Metadata_Dict['Pronunciation_Dict'][word]),
                            self.adjusted_Length_Dict[word],
                            len(self.category_Dict[word, 'Cohort']),
                            len(self.category_Dict[word, 'Rhyme']),
                            len(self.category_Dict[word, 'DAS_Neighborhood']),
                            category,
                            category_Count,
                            not np.isnan(rt_Dict["Onset", "Time_Dependent"])
                            ] + ['{:.5f}'.format(x) for x in category_Flow_Dict[category]]])
                        )

                progress(
                    index + 1,
                    outputs.shape[0],
                    status= result_File
                    )
            print()

        with open(os.path.join(self.result_Path, 'Test', 'RTs.txt').replace('\\', '/'), 'w') as f:
            f.write('\n'.join(reaction_Times))
        with open(os.path.join(self.result_Path, 'Test', 'Category_Flows.txt').replace('\\', '/'), 'w') as f:
            f.write('\n'.join(category_Flows))


    def Data_Generate(self, output, word, identifier, batch_Steps= 200):
        '''
        Data generation is progressed pattern by pattern, not multiple pattern because GPU consuming.
        output: [Steps, Dims]
        '''
        cs_List = []
        for batch_Index in range(0, output.shape[1], batch_Steps):
            cs_List.append(self.Data_Calc(output= output[batch_Index:batch_Index + batch_Steps]))
        cosine_Similarity = np.hstack(cs_List)
        if self.step_Cut:
            cosine_Similarity[:, self.step_Dict[word, identifier]:] = cosine_Similarity[:, [self.step_Dict[word, identifier] - 1]]

        return cosine_Similarity

    @tf.function
    def Data_Calc(self, output):
        '''
        output: [Steps, Dims]
        self.targets: [Num_Words, Dims]
        '''
        output = tf.convert_to_tensor(output, dtype= tf.float32)
        targets = tf.convert_to_tensor(self.targets, dtype= tf.float32)

        tiled_Output = tf.tile(
            tf.expand_dims(output, [0]),
            multiples = [tf.shape(targets)[0], 1, 1]
            )   #[Num_Words, Steps, Dims], increase dimension and tiled for 2D comparing.
        tiled_Targets = tf.tile(
            tf.expand_dims(targets, [1]),
            multiples = [1, tf.shape(output)[0], 1]
            )   #[Num_Words, Steps, Dims], increase dimension and tiled for 2D comparing.
        cosine_Similarity = \
            tf.reduce_sum(tiled_Targets * tiled_Output, axis = 2) / \
            (
                tf.sqrt(tf.reduce_sum(tf.pow(tiled_Targets, 2), axis = 2)) * \
                tf.sqrt(tf.reduce_sum(tf.pow(tiled_Output, 2), axis = 2)) + \
                1e-7
                )  #[Num_Words, Steps]

        return cosine_Similarity

    def RT_Generate(self, word, identifier, data):
        rt_Dict = {
            ('Onset', 'Absolute'): np.nan,
            ('Onset', 'Relative'): np.nan,
            ('Onset', 'Time_Dependent'): np.nan
            }

        target_Index = self.word_Index_Dict[word]
        target_Array = data[target_Index]
        other_Max_Array = np.max(np.delete(data, target_Index, 0), axis=0)  #Target is removed, and using the max value of each time step.

        #Absolute threshold RT
        if not (other_Max_Array > self.absolute_Criterion).any():
            absolute_Check_Array = target_Array > self.absolute_Criterion
            for step in range(self.max_Step):
                if absolute_Check_Array[step]:
                    rt_Dict['Onset', 'Absolute'] = step
                    break

        #Relative threshold RT
        relative_Check_Array = target_Array > (other_Max_Array + self.relative_Criterion)
        for step in range(self.max_Step):
            if relative_Check_Array[step]:
                rt_Dict['Onset', 'Relative'] = step
                break

        #Time dependent RT
        time_Dependency_Check_Array_with_Criterion = target_Array > other_Max_Array + self.time_Dependency_Criterion[1]
        time_Dependency_Check_Array_Sustainment = target_Array > other_Max_Array
        for step in range(self.max_Step - self.time_Dependency_Criterion[0]):
            if all(np.hstack([
                time_Dependency_Check_Array_with_Criterion[step:step + self.time_Dependency_Criterion[0]],
                time_Dependency_Check_Array_Sustainment[step + self.time_Dependency_Criterion[0]:]
                ])):
                rt_Dict['Onset', 'Time_Dependent'] = step
                break

        #Offset_RT = Onset_RT - length
        if not np.isnan(rt_Dict['Onset', 'Absolute']):
            rt_Dict['Offset', 'Absolute'] = rt_Dict['Onset', 'Absolute'] - self.step_Dict[word, identifier]
        else:
            rt_Dict['Offset', 'Absolute'] = rt_Dict['Onset', 'Absolute']    #np.nan
        if not np.isnan(rt_Dict['Onset', 'Relative']):
            rt_Dict['Offset', 'Relative'] = rt_Dict['Onset', 'Relative'] - self.step_Dict[word, identifier]
        else:
            rt_Dict['Offset', 'Relative'] = rt_Dict['Onset', 'Relative']    #np.nan
        if not np.isnan(rt_Dict['Onset', 'Time_Dependent']):
            rt_Dict['Offset', 'Time_Dependent'] = rt_Dict['Onset', 'Time_Dependent'] - self.step_Dict[word, identifier]
        else:
            rt_Dict['Offset', 'Time_Dependent'] = rt_Dict['Onset', 'Time_Dependent']    #np.nan

        return rt_Dict

    def Category_Flow_Generate(self, word, data):   #For categorized flow
        category_Flow_Dict = {}

        for category in ['Target', 'Cohort', 'Rhyme', 'Unrelated']:
            if len(self.category_Dict[word, category]) > 0:
                category_Flow_Dict[category] = np.mean(data[self.category_Dict[word, category],:], axis=0) #Calculation mean of several same category flows.
            else:
                category_Flow_Dict[category] = np.zeros((data.shape[1])) * np.nan   # If there is no word which is belonged a specific category, nan value.

        category_Flow_Dict['All'] = np.mean(data, axis=0)
        category_Flow_Dict['Other_Max'] = np.max(np.delete(data, self.word_Index_Dict[word], 0), axis=0)   #Target is removed, and using the max value of each time step.

        return category_Flow_Dict


    def Pattern_Metadata_Load(self):
        with open(os.path.join(hp_Dict['Pattern']['Pattern_Path'], hp_Dict['Pattern']['Metadata_File']).replace('\\', '/'), 'rb') as f:
            self.pattern_Metadata_Dict = pickle.load(f)

        self.word_Index_Dict = {
            word: index
            for index, (word, _) in enumerate(self.pattern_Metadata_Dict['Target_Dict'].items())
            }

        self.step_Dict = {
            self.pattern_Metadata_Dict['Word_and_Identifier_Dict'][path]: step
            for path, step in self.pattern_Metadata_Dict['Step_Dict'].items()
            }
        self.max_Step = max([step for step in self.step_Dict.values()])

        self.targets = np.array([
            self.pattern_Metadata_Dict['Target_Dict'][word]
            for word, _ in sorted(list(self.word_Index_Dict.items()), key= lambda x: x[1])
            ]).astype(np.float32)

    def Category_Dict_Generate(self):
        self.category_Dict = {}
        for target_Word, target_Pronunciation in self.pattern_Metadata_Dict['Pronunciation_Dict'].items():
            self.category_Dict[target_Word, 'Target'] = []
            self.category_Dict[target_Word, 'Cohort'] = []
            self.category_Dict[target_Word, 'Rhyme'] = []
            self.category_Dict[target_Word, 'DAS_Neighborhood'] = []
            self.category_Dict[target_Word, 'Unrelated'] = []

            for compare_Word, compare_Pronunciation in self.pattern_Metadata_Dict['Pronunciation_Dict'].items():
                compare_Word_Index = self.word_Index_Dict[compare_Word]

                unrelated = True

                if target_Word == compare_Word:
                    self.category_Dict[target_Word, 'Target'].append(compare_Word_Index)
                    unrelated = False
                if target_Pronunciation[0:2] == compare_Pronunciation[0:2] and target_Word != compare_Word: #Cohort
                    self.category_Dict[target_Word, 'Cohort'].append(compare_Word_Index)
                    unrelated = False
                if target_Pronunciation[1:] == compare_Pronunciation[1:] and target_Pronunciation[0] != compare_Pronunciation[0] and target_Word != compare_Word:   #Rhyme
                    self.category_Dict[target_Word, 'Rhyme'].append(compare_Word_Index)
                    unrelated = False
                if unrelated:
                    self.category_Dict[target_Word, 'Unrelated'].append(compare_Word_Index)  #Unrelated
                #For test
                if self.DAS_Neighborhood_Checker(target_Pronunciation, compare_Pronunciation):   #Neighborhood
                    self.category_Dict[target_Word, 'DAS_Neighborhood'].append(compare_Word_Index)

    def DAS_Neighborhood_Checker(self, pronunciation1, pronunciation2):   #Delete, Addition, Substitution neighborhood checking
        #Same pronunciation
        if pronunciation1 == pronunciation2:
            return False

        #Exceed range
        elif abs(len(pronunciation1) - len(pronunciation2)) > 1:    #The length difference is bigger than 1, two pronunciations are not related.
            return False

        #Deletion
        elif len(pronunciation1) == len(pronunciation2) + 1:
            for index in range(len(pronunciation1)):
                deletion = pronunciation1[:index] + pronunciation1[index + 1:]
                if deletion == pronunciation2:
                    return True

        #Addition
        elif len(pronunciation1) == len(pronunciation2) - 1:
            for index in range(len(pronunciation2)):
                deletion = pronunciation2[:index] + pronunciation2[index + 1:]
                if deletion == pronunciation1:
                    return True

        #Substitution
        elif len(pronunciation1) == len(pronunciation2):
            for index in range(len(pronunciation1)):
                pronunciation1_Substitution = pronunciation1[:index] + pronunciation1[index + 1:]
                pronunciation2_Substitution = pronunciation2[:index] + pronunciation2[index + 1:]
                if pronunciation1_Substitution == pronunciation2_Substitution:
                    return True

        return False

    def Adjusted_Length_Dict_Generate(self): #For uniqueness point.
        self.adjusted_Length_Dict = {}

        for word, pronunciation in self.pattern_Metadata_Dict['Pronunciation_Dict'].items():
            for cut_Length in range(1, len(pronunciation) + 1):
                cut_Pronunciation = pronunciation[:cut_Length]
                cut_Comparer_List = [comparer[:cut_Length] for comparer in self.pattern_Metadata_Dict['Pronunciation_Dict'].values() if pronunciation != comparer]
                if not cut_Pronunciation in cut_Comparer_List:  #When you see a part of target phoneme string, if there is no other competitor.
                    self.adjusted_Length_Dict[word] = cut_Length - len(pronunciation) - 1
                    break
            if not word in self.adjusted_Length_Dict.keys():
                self.adjusted_Length_Dict[word] = 0


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-d', '--directory', default= '', type= str)
    argParser.add_argument('-a', '--absolute', default= 0.7, type= float)
    argParser.add_argument('-r', '--relative', default= 0.05, type= float)
    argParser.add_argument('-tw', '--time_dependency_width', default= 10, type= float)
    argParser.add_argument('-th', '--time_dependency_height', default= 0.05, type= float)
    argument_Dict = vars(argParser.parse_args())

    new_Analyzer = Analyzer(
        path= argument_Dict['directory'],
        absolute_Criterion= argument_Dict['absolute'],
        relative_Criterion= argument_Dict['relative'],
        time_Dependency_Criterion= (
            argument_Dict['time_dependency_width'],
            argument_Dict['time_dependency_height']
            ),
        step_Cut= True
        )
