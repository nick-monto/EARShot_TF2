import numpy as np
import json, os, time, librosa, pickle
from collections import deque, Sequence
from threading import Thread
from random import shuffle, randint

from Audio import spectrogram, melspectrogram
from ProgressBar import progress

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Feeder:
    def __init__(
        self,
        start_Epoch,
        is_Training= False,
        excluded_Identifier= None
        ):
        self.is_Training = is_Training
        self.start_Epoch = start_Epoch
        self.Pattern_Metadata_Load()
        
        
        if isinstance(excluded_Identifier, str):
            self.excluded_Identifier_List = [excluded_Identifier.upper()]
        elif isinstance(excluded_Identifier, Sequence): # When multiple excluded identifier used. But currently this function is not being used.
            self.excluded_Identifier_List = [identifier.upper() for identifier in excluded_Identifier]
        else:
            self.excluded_Identifier_List = []

        if start_Epoch > 0:
            self.Exclusion_Info_Dict_Load()            
        else:
            self.Exclusion_Info_Dict_Generate()

        print('Pattern counts by exclusion type')
        for exclusion_Type in ['Training', 'Pattern_Excluded', 'Identifier_Excluded', 'Test_Only']:
            print('{}: {}'.format(exclusion_Type, len(self.pattern_Path_Dict[exclusion_Type])))

        if self.is_Training and start_Epoch >= hp_Dict['Train']['Max_Epoch_without_Exclusion']:
            print('WARNING: The start epoch is greater than or equal to maximum epoch. Training mode turned off.')
            self.is_Training = False
            
        if self.is_Training:
            self.is_Finished = False
            self.Pattern_Metadata_Load()
            self.pattern_Queue = deque()
            pattern_Generate_Thread = Thread(
                target= self.Pattern_Generate
                )
            pattern_Generate_Thread.daemon = True
            pattern_Generate_Thread.start()
        else:
            self.is_Finished = True

    def Pattern_Metadata_Load(self):
        with open(os.path.join(hp_Dict['Pattern']['Pattern_Path'], hp_Dict['Pattern']['Metadata_File']).replace("\\", "/"), "rb") as f:
            self.pattern_Metadata_Dict = pickle.load(f)

    def Exclusion_Info_Dict_Generate(self):
        '''
        When 'Exclusion_Mode' is 'P'(Pattern based), each identifier's partial pattern will not be trained.
        When 'Exclusion_Mode' is 'I'(Identifier based), a identifier's all pattern will not be trained.        
        When 'Exclusion_Mode' is 'M'(Mix based), each identifier's partial pattern will not be trained and a identifier's all pattern will not be trained.
        When 'Exclusion_Mode' is ''(None), all pattern will be trained.

        The identifiers who are in hyper parameter 'Test_Only_Identifier_List' are always excluded.
        '''
        self.pattern_Path_Dict = {
            'Training': {}, # Always train
            'Pattern_Excluded': {}, # Only trained when epoch is between 'with' and 'without' exclusion.
            'Identifier_Excluded': {},  # Only trained when epoch is between 'with' and 'without' exclusion.
            'Test_Only': {},    # Always not trained.
            }
        
        self.pattern_Path_Dict['Test_Only'].update({
            (word, identifier): path
            for (word, identifier), path in self.pattern_Metadata_Dict["Pattern_Path_Dict"].items()
            if identifier in hp_Dict['Train']['Test_Only_Identifier_List']
            })        

        if hp_Dict['Train']['Exclusion_Mode'] is None:
            self.pattern_Path_Dict['Training'].update({
                (word, identifier): path
                for (word, identifier), path in self.pattern_Metadata_Dict["Pattern_Path_Dict"].items()
                if not identifier in hp_Dict['Train']['Test_Only_Identifier_List']
                })
            return

        # Shuffle identifier and word
        identifier_List = list(set([identifier for _, identifier in self.pattern_Metadata_Dict["Pattern_Path_Dict"].keys()]))
        word_List = list(set([word for word, _ in self.pattern_Metadata_Dict["Pattern_Path_Dict"].keys()]))
        for excluded_Identifier in hp_Dict['Train']['Test_Only_Identifier_List']:
            identifier_List.remove(excluded_Identifier)            
        shuffle(identifier_List)
        shuffle(word_List)
        
        if hp_Dict['Train']['Exclusion_Mode'].upper() == 'P':   # Word excluded
            excluded_Size = len(word_List) // len(identifier_List)
            for index, identifier in enumerate(identifier_List):
                for word in word_List[:index * excluded_Size] + word_List[(index + 1) * excluded_Size:]:
                    self.pattern_Path_Dict['Training'][word, identifier] = self.pattern_Metadata_Dict["Pattern_Path_Dict"][word, identifier]
                for word in word_List[index * excluded_Size:(index + 1) * excluded_Size]:
                    self.pattern_Path_Dict['Pattern_Excluded'][word, identifier] = self.pattern_Metadata_Dict["Pattern_Path_Dict"][word, identifier]
            return
        
        for excluded_Identifier in self.excluded_Identifier_List:
            if not excluded_Identifier in identifier_List:
                raise ValueError('Identifier \'{}\' is not included in the pattern data.'.format(excluded_Identifier))

        #Selete excluded identifier when it is None
        if len(self.excluded_Identifier_List) == 0:
            self.excluded_Identifier_List = [identifier_List[-1]]

        #If excluded identifier is not determined, select one identifier
        for excluded_Identifier in self.excluded_Identifier_List:
            identifier_List.remove(excluded_Identifier)

        if hp_Dict['Train']['Exclusion_Mode'].upper() == 'I':
            for word in word_List:
                for identifier in identifier_List:
                    self.pattern_Path_Dict['Training'][word, identifier] = self.pattern_Metadata_Dict["Pattern_Path_Dict"][word, identifier]
                for identifier in self.excluded_Identifier_List:
                    self.pattern_Path_Dict['Identifier_Excluded'][word, identifier] = self.pattern_Metadata_Dict["Pattern_Path_Dict"][word, identifier]
            return

        if hp_Dict['Train']['Exclusion_Mode'].upper() == 'M':
            excluded_Size = len(word_List) // len(identifier_List)  #Determine the ratio of excluded word. But the size is different from 'P' mode because of identifier exclusion.
            for index, identifier in enumerate(identifier_List):    #Pattern exclusion
                for word in word_List[:index * excluded_Size] + word_List[(index + 1) * excluded_Size:]:
                    self.pattern_Path_Dict['Training'][word, identifier] = self.pattern_Metadata_Dict["Pattern_Path_Dict"][word, identifier]
                for word in word_List[index * excluded_Size:(index + 1) * excluded_Size]:
                    self.pattern_Path_Dict['Pattern_Excluded'][word, identifier] = self.pattern_Metadata_Dict["Pattern_Path_Dict"][word, identifier]
            for word in word_List:  #Identifier exclusion
                for identifier in self.excluded_Identifier_List:
                    self.pattern_Path_Dict['Identifier_Excluded'][word, identifier] = self.pattern_Metadata_Dict["Pattern_Path_Dict"][word, identifier]
            return

        raise ValueError('Unsupported exclusion mode \'{}\'. Mode must be one of [P, I, M, None(null)].'.format(hp_Dict['Train']['Exclusion_Mode'].upper()))

    def Exclusion_Info_Dict_Load(self):
        training_Metadata_File= os.path.join(hp_Dict['Result_Path'], 'Training_Metadta.pickle').replace("\\", "/")

        with open(training_Metadata_File, 'rb') as f:
            self.pattern_Path_Dict = pickle.load(f)
        
        word_List = list(set([word for word, _ in self.pattern_Metadata_Dict["Pattern_Path_Dict"].keys()]))
        identifier_List = list(set([identifier for _, identifier in self.pattern_Metadata_Dict["Pattern_Path_Dict"].keys()]))
        path_List = list(set([path for path in self.pattern_Metadata_Dict["Pattern_Path_Dict"].values()]))

        for path_Dict in self.pattern_Path_Dict.values():
            for (word, identifier), path in path_Dict.items():
                if not word in word_List:
                    raise ValueError('Word \'{}\' is not in pattern metadata. Check the pattern hyper parameters.'.format(word))
                if not identifier in identifier_List:
                    raise ValueError('Identifier \'{}\' is not in pattern metadata. Check the pattern hyper parameters.'.format(identifier))
                if not path in path_List:
                    raise ValueError('Path \'{}\' is not in pattern metadata. Check the pattern hyper parameters.'.format(path))

    def Pattern_Generate(self):
        pattern_Cache_Dict = {}
        epoch = self.start_Epoch
        
        while True:
            pattern_Path_List = []
            pattern_Path_List.extend([path for path in self.pattern_Path_Dict['Training'].values()])
            if epoch < hp_Dict['Train']['Max_Epoch_with_Exclusion']:
                pass    
            elif epoch < hp_Dict['Train']['Max_Epoch_without_Exclusion']:
                pattern_Path_List.extend([path for path in self.pattern_Path_Dict['Pattern_Excluded'].values()])
                pattern_Path_List.extend([path for path in self.pattern_Path_Dict['Identifier_Excluded'].values()])
            else:
                self.is_Finished = True
                break

            shuffle(pattern_Path_List)
            pattern_Batch_List = [  # Split pattern list to genrate batchs
                pattern_Path_List[x:x + hp_Dict['Train']['Batch_Size']]
                for x in range(0, len(pattern_Path_List), hp_Dict['Train']['Batch_Size'])
                ]

            current_Index = 0
            is_New_Epoch = True            
            while current_Index < len(pattern_Batch_List):
                #If queue is full, pattern generating is stopped while 0.1 sec.
                if len(self.pattern_Queue) >= hp_Dict['Train']['Max_Queue']:
                    time.sleep(0.1)
                    continue

                pattern_Batch = pattern_Batch_List[current_Index]
                
                acoustics = []
                semantics = []
                acoustic_Steps = []

                for path in pattern_Batch:
                    if path in pattern_Cache_Dict.keys():
                        pattern_Dict = pattern_Cache_Dict[path]
                    else:
                        with open(os.path.join(hp_Dict['Pattern']['Pattern_Path'], path).replace("\\", "/"), "rb") as f:
                            pattern_Dict = pickle.load(f)

                    acoustics.append(pattern_Dict['Acoustic'])
                    semantics.append(pattern_Dict['Semantic'])
                    acoustic_Steps.append(pattern_Dict['Acoustic'].shape[0])

                    if hp_Dict['Train']['Use_Pattern_Cache']:
                        pattern_Cache_Dict[path] = pattern_Dict

                acoustics = self.Force_Pattern_Stack(acoustics).astype(np.float32)
                semantics = np.stack(semantics, axis= 0).astype(np.float32)
                acoustic_Steps = np.stack(acoustic_Steps, axis= 0).astype(np.int32)

                self.pattern_Queue.append([
                    epoch,
                    is_New_Epoch,
                    {
                        'acoustics': acoustics,
                        'acoustic_Steps': acoustic_Steps,
                        'semantics': semantics,                        
                        }
                    ])
                current_Index += 1
                is_New_Epoch = False
            epoch += 1
    
    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01)
        return self.pattern_Queue.popleft()


    def Get_Test_Pattern(self):
        pattern_Info_List = []
        pattern_Info_List.extend([(word, identifier, 'Training') for word, identifier in self.pattern_Path_Dict['Training'].keys()])
        pattern_Info_List.extend([(word, identifier, 'Pattern_Excluded') for word, identifier in self.pattern_Path_Dict['Pattern_Excluded'].keys()])
        pattern_Info_List.extend([(word, identifier, 'Identifier_Excluded') for word, identifier in self.pattern_Path_Dict['Identifier_Excluded'].keys()])
        pattern_Info_List.extend([(word, identifier, 'Test_Only') for word, identifier in self.pattern_Path_Dict['Test_Only'].keys()])

        pattern_Path_List = []
        pattern_Path_List.extend([path for path in self.pattern_Path_Dict['Training'].values()])
        pattern_Path_List.extend([path for path in self.pattern_Path_Dict['Pattern_Excluded'].values()])
        pattern_Path_List.extend([path for path in self.pattern_Path_Dict['Identifier_Excluded'].values()])
        pattern_Path_List.extend([path for path in self.pattern_Path_Dict['Test_Only'].values()])

        patterns = {}
        for index, path in enumerate(pattern_Path_List):
            with open(os.path.join(hp_Dict['Pattern']['Pattern_Path'], path).replace("\\", "/"), "rb") as f:
                patterns[path] = pickle.load(f)
            progress(
                index + 1,
                len(pattern_Path_List),
                status='Test pattern loading'
                )
        print()
        max_Step = max([pattern['Acoustic'].shape[0] for pattern in patterns.values()])

        pattern_Batch_List = [  # Split pattern list to genrate batchs
            pattern_Path_List[x:x + hp_Dict['Train']['Batch_Size']]
            for x in range(0, len(pattern_Path_List), hp_Dict['Train']['Batch_Size'])
            ]

        test_Pattern_List = []
        for pattern_Batch in pattern_Batch_List:            
            acoustics = []
            # semantics = []
            # acoustic_Steps = []

            for path in pattern_Batch:                
                pattern_Dict = patterns[path]
                acoustics.append(pattern_Dict['Acoustic'])
                # semantics.append(pattern_Dict['Semantic'])
                # acoustic_Steps.append(pattern_Dict['Acoustic'].shape[0])

            acoustics = self.Force_Pattern_Stack(acoustics, max_Step= max_Step).astype(np.float32)
            # semantics = np.stack(semantics, axis= 0).astype(np.float32)
            # acoustic_Steps = np.stack(acoustic_Steps, axis= 0).astype(np.int32)

            test_Pattern_List.append({
                'acoustics': acoustics,
                # 'acoustic_Steps': acoustic_Steps,
                # 'semantics': semantics,                
                })

        return pattern_Info_List, test_Pattern_List

    def Get_Test_Pattern_from_Wav(self, wav_Path_List):
        patterns = {}
        for path in wav_Path_List:
            if hp_Dict['Pattern']['Acoustic']['Mode'].upper() == "Spectrogram".upper():
                sig = librosa.core.load(
                    path,
                    sr = hp_Dict['Pattern']['Acoustic']['Spectrogram']['Sample_Rate']
                    )[0]
                sig = librosa.effects.trim( # Trim
                    sig,
                    frame_length = 32,
                    hop_length=16
                    )[0]
                spec = spectrogram(
                    sig,
                    num_freq= hp_Dict['Pattern']['Acoustic']['Spectrogram']['Dimension'],
                    frame_shift_ms= hp_Dict['Pattern']['Acoustic']['Spectrogram']['Frame_Shift'],
                    frame_length_ms= hp_Dict['Pattern']['Acoustic']['Spectrogram']['Frame_Length'],
                    sample_rate= hp_Dict['Pattern']['Acoustic']['Spectrogram']['Sample_Rate'],
                    )
                patterns[path] = np.transpose(spec).astype(np.float32)    # [Time, Dim]
            elif hp_Dict['Pattern']['Acoustic']['Mode'].upper() == "Mel".upper():
                sig = librosa.core.load(
                    path,
                    sr = hp_Dict['Pattern']['Acoustic']['Mel']['Sample_Rate']
                    )[0]
                sig = librosa.effects.trim(   #Trim
                    sig,
                    frame_length = 32,
                    hop_length=16
                    )[0]
                mel = melspectrogram(
                    sig,
                    num_freq= hp_Dict['Pattern']['Acoustic']['Mel']['Spectrogram_Dimension'],
                    frame_shift_ms= hp_Dict['Pattern']['Acoustic']['Mel']['Frame_Shift'],
                    frame_length_ms= hp_Dict['Pattern']['Acoustic']['Mel']['Frame_Length'],
                    num_mels= hp_Dict['Pattern']['Acoustic']['Mel']['Dimension'],
                    sample_rate= hp_Dict['Pattern']['Acoustic']['Mel']['Sample_Rate'],
                    max_abs_value= hp_Dict['Pattern']['Acoustic']['Mel']['Max_Abs']
                    )
                patterns[path] = np.transpose(mel).astype(np.float32)

        max_Step = max([pattern.shape[0] for pattern in patterns.values()])

        pattern_Batch_List = [  # Split pattern list to genrate batchs
            wav_Path_List[x:x + hp_Dict['Train']['Batch_Size']]
            for x in range(0, len(wav_Path_List), hp_Dict['Train']['Batch_Size'])
            ]

        test_Pattern_List = []
        for pattern_Batch in pattern_Batch_List:
            acoustics = []
            # acoustic_Steps = []

            for path in pattern_Batch:                
                pattern_Dict = patterns[path]
                acoustics.append(pattern_Dict['Acoustic'])
                # acoustic_Steps.append(pattern_Dict['Acoustic'].shape[0])

            acoustics = self.Force_Pattern_Stack(acoustics, max_Step= max_Step).astype(np.float32)
            # acoustic_Steps = np.stack(acoustic_Steps, axis= 0).astype(np.int32)

            test_Pattern_List.append({
                'acoustics': acoustics,
                # 'acoustic_Steps': acoustic_Steps,
                })

        return test_Pattern_List

    def Force_Pattern_Stack(self, pattern_List, max_Step = None):
        max_Step = max_Step or max([pattern.shape[0] for pattern in pattern_List])
        pattern_List = [
            np.concatenate(
                [pattern, np.zeros((max_Step - pattern.shape[0], pattern.shape[1]), dtype= pattern.dtype)]
                )
            for pattern in pattern_List
            ]

        return np.stack(pattern_List, axis= 0)


if __name__ == "__main__":
    new_Feeder = Feeder(0, True)
    x = new_Feeder.Get_Test_Pattern()    
    print(x)
    print(len(x))
    assert False
    while True:
        time.sleep(1.0)
        print(new_Feeder.Get_Pattern())
        print(len(new_Feeder.pattern_Queue))