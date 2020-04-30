import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
import os, argparse, pickle, json

from Model import EARShot
from ProgressBar import progress

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

if not hp_Dict['Device'] is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp_Dict['Device']

if hp_Dict['Use_Mixed_Precision']:    
    policy = mixed_precision.Policy('mixed_float16')
else:
    policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

class Hidden_Analyzer:
    def __init__(self, epoch):
        self.epoch = epoch

        self.Feature_Load()
        self.Wav_Load()
        self.Model_Load()

        self.Activation_Dict_Generate()        

    def Feature_Load(self):
        with open(hp_Dict['Hidden_Analysis']['Phoneme_Feature'], 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.feature_List = lines[0].upper().strip().split("\t")[2:] #Feature name list
        self.index_feature_Name_Dict = {
            index: feature_Name.strip()
            for index, feature_Name in enumerate(self.feature_List)
            }  # Index to feature name matching
        self.feature_Name_Index_Dict = {
            feature_Name.strip(): index
            for index, feature_Name in self.index_feature_Name_Dict.items()
            }  # Feature name to Index matching

        #Phoneme list and feature dict
        self.phoneme_Label_Dict = {} #key, value: CMU code, IPA
        self.consonant_List = []
        self.vowel_List = []
        self.feature_Dict = {
            feature_Name: []
            for feature_Name in self.feature_List
            }

        for line in lines[1:]:
            line = line.strip().split("\t")
            self.phoneme_Label_Dict[line[0]] = line[1]   #CMU code to IPA matching
            
            #Checking consonant or vowel
            if line[2] == "1":
                self.consonant_List.append(line[0])
            elif line[2] == "0":
                self.vowel_List.append(line[0])
            
            # Checking each features have which phoneme
            for index, value in enumerate([int(feature.strip()) for feature in line[2:]]):
                if value == 1:
                    self.feature_Dict[self.index_feature_Name_Dict[index]].append(line[0])

        self.phone_List = self.consonant_List + self.vowel_List

        #Generating diphone
        self.diphone_Type_List = ["CV", "VC"]
        self.diphone_List = []
        self.diphone_List.extend([
            consonant + vowel
            for consonant in self.consonant_List for vowel in self.vowel_List
            ])
        self.diphone_List.extend([
            vowel + consonant
            for consonant in self.consonant_List for vowel in self.vowel_List
            ])

        #CMU code to IPA matching
        diphone_Label_Dict = {}
        for consonant in self.consonant_List:
            for vowel in self.vowel_List:
                diphone_Label_Dict[consonant + vowel] = self.phoneme_Label_Dict[consonant] + self.phoneme_Label_Dict[vowel]
                diphone_Label_Dict[vowel + consonant] = self.phoneme_Label_Dict[vowel] + self.phoneme_Label_Dict[consonant]

    def Wav_Load(self):
        self.identifier_List = []
        for _, _, files in os.walk(hp_Dict['Hidden_Analysis']['Diphone_Wav_Path']):
            for file in files:
                self.identifier_List.append(os.path.splitext(file)[0].strip().split('_')[2].upper())
        self.identifier_List = sorted(list(set(self.identifier_List)))

        self.wav_Path_List = []
        self.wav_Path_Index_Dict = {}

        self.phone_File_List_Dict = {
            (phone, identifier): []
            for phone in self.phone_List
            for identifier in (['ALL'] if hp_Dict['Hidden_Analysis']['Only_All'] else (self.identifier_List + ['ALL']))
            }
        self.feature_File_List_Dict = {
            (feature, identifier): []
            for feature in self.feature_List
            for identifier in (['ALL'] if hp_Dict['Hidden_Analysis']['Only_All'] else (self.identifier_List + ['ALL']))
            }
        for root, _, files in os.walk(hp_Dict['Hidden_Analysis']['Diphone_Wav_Path']):
            for file in files:
                file = file.upper()
                diphone_Type, diphone, identifier = os.path.splitext(file)[0].strip().split('_')
                if not (diphone_Type in self.diphone_Type_List and diphone in self.diphone_List):
                    continue

                path = os.path.join(root, file).replace("\\", "/")
                self.wav_Path_Index_Dict[path] = len(self.wav_Path_List)
                self.wav_Path_List.append(path)

                #Phone
                if diphone_Type == "CV":    #If file is CV, first phoneme is a consonant and second phoneme is a vowel.
                    consonant = diphone[:-2]
                    vowel = diphone[-2:]
                    self.phone_File_List_Dict[consonant, 'ALL'].append(path)
                    if not hp_Dict['Hidden_Analysis']['Only_All']:
                        self.phone_File_List_Dict[consonant, identifier].append(path)

                    for feature, feature_Phoneme_List in self.feature_Dict.items():
                        if consonant in feature_Phoneme_List:  #Checking the phoneme is related to the feature.
                            self.feature_File_List_Dict[feature, 'ALL'].append(path)
                            if not hp_Dict['Hidden_Analysis']['Only_All']:
                                self.feature_File_List_Dict[feature, identifier].append(path)
                elif diphone_Type == "VC":  #If file is VC, first phoneme is a vowel and second phoneme is a consonant.
                    vowel = diphone[:2]
                    consonant = diphone[2:]
                    self.phone_File_List_Dict[vowel, 'ALL'].append(path)
                    if not hp_Dict['Hidden_Analysis']['Only_All']:
                        self.phone_File_List_Dict[vowel, identifier].append(path)

                    for feature, feature_Phoneme_List in self.feature_Dict.items():
                        if vowel in feature_Phoneme_List:  #Checking the phoneme is related to the feature.
                            self.feature_File_List_Dict[feature, 'ALL'].append(path)
                            if not hp_Dict['Hidden_Analysis']['Only_All']:
                                self.feature_File_List_Dict[feature, identifier].append(path)

    def Model_Load(self):
        self.model = EARShot(start_Epoch= self.epoch, is_Training= False)
        self.model.Restore()


    def Activation_Dict_Generate(self):
        pattern_List = self.model.feeder.Get_Test_Pattern_from_Wav(
            wav_Path_List= self.wav_Path_List
            )

        hiddens = []
        for index, pattern_Dict in enumerate(pattern_List):
            hiddens.append(self.model.Hidden_Step(**pattern_Dict))
            progress(
                index + 1,
                len(pattern_List),
                status= 'Activation dict generating...'
                )
        print()
        hiddens = np.vstack(hiddens)    # [File_Nums, Time, Dim]

        self.activation_Dict = {
            'Phone': {    # Gathering by phoneme
                (phone, identifier): hiddens[[self.wav_Path_Index_Dict[path] for path in path_List]]    # [File_Nums, Time, Dim]
                for (phone, identifier), path_List in self.phone_File_List_Dict.items()
                },
            'Feature': {  # Gathering by feature
                (feature, identifier): hiddens[[self.wav_Path_Index_Dict[path] for path in path_List]]  # [File_Nums, Time, Dim]
                for (feature, identifier), path_List in self.feature_File_List_Dict.items()
                }
            }

    def Export_Flow(self):
        self._Flow_Dict_Generate()

        phone_Flow_Path = os.path.join(hp_Dict['Result_Path'], 'Hidden', 'Flow', 'Phone', 'TXT').replace('\\', '/')
        os.makedirs(phone_Flow_Path, exist_ok= True)
        for identifier, flows in self.flow_Dict['Phone'].items():
            for unit_Index, unit_Flows in enumerate(flows):
                export_List = ['\t'.join(['Phone'] + ['{}'.format(flow_Index) for flow_Index in range(unit_Flows.shape[1])])]
                for phone, phone_Flows in zip(self.phone_List, unit_Flows):
                    export_List.append('\t'.join([self.phoneme_Label_Dict[phone]] + ['{:.5f}'.format(x) for x in phone_Flows]))
                
                file_Name = '{}.U_{}.I_{}.txt'.format('Phone', unit_Index, identifier)
                with open(os.path.join(phone_Flow_Path, file_Name).replace('\\', '/'), 'w', encoding='utf-8') as f:
                    f.write("\n".join(export_List))

        feature_Flow_Path = os.path.join(hp_Dict['Result_Path'], 'Hidden', 'Flow', 'Feature', 'TXT').replace('\\', '/')
        os.makedirs(feature_Flow_Path, exist_ok= True)
        for identifier, flows in self.flow_Dict['Feature'].items():
            for unit_Index, unit_Flows in enumerate(flows):
                export_List = ['\t'.join(['Feature'] + ['{}'.format(flow_Index) for flow_Index in range(unit_Flows.shape[1])])]
                for feature, feature_Flows in zip(self.feature_List, unit_Flows):
                    export_List.append('\t'.join([feature] + ['{:.5f}'.format(x) for x in feature_Flows]))
                
                file_Name = '{}.U_{:04d}.I_{}.txt'.format('Feature', unit_Index, identifier)
                with open(os.path.join(feature_Flow_Path, file_Name).replace('\\', '/'), 'w', encoding='utf-8') as f:
                    f.write("\n".join(export_List))

    def _Flow_Dict_Generate(self):
        self.flow_Dict = {
            'Phone': {},
            'Feature': {}
            }

        identifiers = ['ALL'] if hp_Dict['Hidden_Analysis']['Only_All'] else (self.identifier_List + ['ALL'])
        for index, identifier in enumerate(identifiers):
            self.flow_Dict['Phone'][identifier] = np.stack([
                np.transpose(np.mean(self.activation_Dict['Phone'][phone, identifier], axis= 0))    # [File_Nums, Time, Dim] -> [Time, Dim] -> [Dim, Time]
                for phone in self.phone_List
                ], axis= 1) # [Dim, Phone_Nums, Time]

            self.flow_Dict['Feature'][identifier] = np.stack([
                np.transpose(np.mean(self.activation_Dict['Feature'][feature, identifier], axis= 0))    # [File_Nums, Time, Dim] -> [Time, Dim] -> [Dim, Time]
                for feature in self.feature_List
                ], axis= 1) # [Dim, Feature_Nums, Time]

            progress(index + 1, len(identifiers), status= 'Flow generating...')
        print()


    def Export_Map(self):
        self._Map_Dict_Generate()

        psi_Map_Path = os.path.join(hp_Dict['Result_Path'], 'Hidden', 'Map', 'PSI', 'TXT').replace('\\', '/')
        os.makedirs(psi_Map_Path, exist_ok= True)
        for (identifier, criterion), maps in self.map_Dict['Phone'].items():
            export_List = ['\t'.join(['Phone'] + ['{}'.format(unit_Index) for unit_Index in range(maps.shape[1])])]
            for phone, phone_Maps in zip(self.phone_List, maps):
                export_List.append('\t'.join([self.phoneme_Label_Dict[phone]] + ['{}'.format(x) for x in phone_Maps]))

            file_Name = '{}.C_{:.2f}.I_{}.txt'.format('PSI', criterion, identifier)
            with open(os.path.join(psi_Map_Path, file_Name).replace('\\', '/'), 'w', encoding='utf-8') as f:
                    f.write("\n".join(export_List))


        fsi_Map_Path = os.path.join(hp_Dict['Result_Path'], 'Hidden', 'Map', 'FSI', 'TXT').replace('\\', '/')
        os.makedirs(fsi_Map_Path, exist_ok= True)
        for (identifier, criterion), maps in self.map_Dict['Feature'].items():
            export_List = ['\t'.join(['Feature'] + ['{}'.format(unit_Index) for unit_Index in range(maps.shape[1])])]
            for feature, feature_Maps in zip(self.feature_List, maps):
                export_List.append('\t'.join([feature] + ['{}'.format(x) for x in feature_Maps]))

            file_Name = '{}.C_{:.2f}.I_{}.txt'.format('FSI', criterion, identifier)
            with open(os.path.join(fsi_Map_Path, file_Name).replace('\\', '/'), 'w', encoding='utf-8') as f:
                    f.write("\n".join(export_List))

    def _Map_Dict_Generate(self):
        self.map_Dict = {
            'Phone': {},
            'Feature': {}
            }

        identifiers = ['ALL'] if hp_Dict['Hidden_Analysis']['Only_All'] else (self.identifier_List + ['ALL'])
        criteria = np.arange(
            start= hp_Dict['Hidden_Analysis']['Sensitivity_Index_Criteria'][0],
            stop= hp_Dict['Hidden_Analysis']['Sensitivity_Index_Criteria'][1] + hp_Dict['Hidden_Analysis']['Sensitivity_Index_Criteria'][2],
            step= hp_Dict['Hidden_Analysis']['Sensitivity_Index_Criteria'][2],
            )

        progress_Index = 0
        for identifier in identifiers:
            averages = np.stack([
                np.mean(self.activation_Dict['Phone'][phone, identifier], axis= (0, 1)) # [File_Nums, Time, Dim] -> [Dim]
                for phone in self.phone_List
                ], axis= 1) # [Dim, Phone_Nums]
            for criterion in criteria:
                self.map_Dict['Phone'][identifier, criterion] = np.stack([
                    self._Map_Calc(x, criterion) # [Phone_Nums]
                    for x in averages
                    ], axis= 1)  # [Phone_Nums, Dim]

                progress_Index += 1
                progress(progress_Index, len(identifiers) * len(criteria) * 2, status= 'Map generating...')
                

        for identifier in identifiers:
            averages = np.stack([
                np.mean(self.activation_Dict['Feature'][feature, identifier], axis= (0, 1)) # [File_Nums, Time, Dim] -> [Dim]
                for feature in self.feature_List
                ], axis= 1) # [Dim, Feature_Nums]
            for criterion in criteria:
                self.map_Dict['Feature'][identifier, criterion] = np.stack([
                    self._Map_Calc(x, criterion) # [Feature_Nums]
                    for x in averages
                    ], axis= 1)  # [Feature_Nums, Dim]

                progress_Index += 1
                progress(progress_Index, len(identifiers) * len(criteria) * 2, status= 'Map generating...')

        print()

    @tf.function
    def _Map_Calc(self, averages, criterion):
        averages = tf.convert_to_tensor(averages, dtype= tf.as_dtype(policy.compute_dtype))
        criterion = tf.convert_to_tensor(criterion, dtype= tf.as_dtype(policy.compute_dtype))

        tiled_Averages = tf.tile( #The activation array is tiled for 2D calculation.
            tf.expand_dims(averages, axis=1),
            multiples=[1, tf.shape(averages)[0]]
            )
    
        #Over criterion, getting 1 point.
        sensitivity_Maps = tf.sign(
            tf.clip_by_value(tiled_Averages - (tf.transpose(tiled_Averages) + criterion), 0, 1)
            )   # [Phoneme, Phoneme], Comparing each other phonemes by positive direction (Negative becomes 0).
        sensitivity_Indices = tf.reduce_sum(sensitivity_Maps, axis=1)   # [Phoneme], Sum score

        return sensitivity_Indices


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-e', '--epoch', required= True, type= int)
    argument_Dict = vars(argParser.parse_args())

    new_Hidden_Analyzer = Hidden_Analyzer(epoch = argument_Dict['epoch'])
    new_Hidden_Analyzer.Export_Flow()
    new_Hidden_Analyzer.Export_Map()