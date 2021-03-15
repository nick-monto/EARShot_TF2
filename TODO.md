# EARSHOT TODO

## People (for purposes of putative task assignment):

  - JM (Jim Magnuson)
  - KB (Kevin Brown)
  - CB (Christian Broadbeck)
  - NM (Nick Monto)
  - ALL (Everyone)

## Tasks:

(Completed tasks should be routinely purged)

  - better use of parameters structures in training/analysis scripts
  - fix analysis so it can use all checkpoint files

## Under Discussion:

(These items should either be moved into Tasks or deleted)

  - more flexibility in input patterns; have the model auto-sniff input/output sizes from a pattern file

## Notes

(Space for notes on the code that don't fit neatly into the categories above)

  - parameters from the .json file that `Pattern_Generator` uses:
    - only uses parameters from the 'Pattern' group
  - parameters that `Feeder` uses:
    - 'Result_Path', 'Pattern -> Pattern_Path', 'Pattern' -> Metadata_File'
    - 'Train' -> 'Test_Only_Identifier_List', 'Train' -> 'Exclusion Mode'
    - 'Train' -> 'Batch_Size', 'Train' -> 'Max_Queue', 'Train' -> 'Use_Pattern_Cache'
    - 'Pattern' -> 'Acoustic' -> etc.
  - parameters that `Modules` uses:
    - 'Model' -> 'Prenet' -> etc., 'Model' -> 'Hidden' -> etc.
    - 'Pattern' -> 'Semantic' -> etc.
  - parameters that `Model` uses:
    - 'Pattern' -> etc., 'Model' -> 'Hidden' -> etc.
    - 'Train' -> 'Loss', 'Model' -> 'Prenet' -> etc.
    - 'Train' -> 'Learning_Rate' -> etc., 'Train' -> 'ADAM' -> etc.
    - 'Train' -> 'SGD' -> etc., 'Train' -> 'Checkpoint_Save_Timing'
    - 'Train' -> 'Test_Timing', 'Train' -> 'Batch_Size'
    - 'Train' -> 'Max_Epoch_without_Exclusion'
