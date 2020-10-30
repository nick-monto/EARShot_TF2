# EARSHOT TODO

## People (for purposes of putative task assignment):

  - JM (Jim Magnuson)
  - KB (Kevin Brown)
  - CB (Christian Broadbeck)
  - NM (Nick Monto)
  - ALL (Everyone)

## Tasks:

(Completed tasks should be routinely purged)

  - use `pathlib` to avoid having to replace //
  - try to comment out device lines to see if we can utilize > 1 GPU
  - convert to `python` module (this is happening - KB)
  - better/standardized formatting (CB; KB is doing this as he does the modularization)
  - `excluded_Identifier`: can you use it to leave out a talker AND/OR a word (e.g. leave out all of Ava's items, or all instances of YELP, or AVA_YELP)?
  - add docstrings to methods/classes (doing this as we rewrite to make a module)
  - `sphinx` for automatic documentation generation (CB)
  - add everyone to R drive or lab NAS for big file storage (JM)
  - set long-term goals, e.g. commenting, keras-izing (ALL)
  - items for Gaskell/Marslen-Wilson semantic blends (JM)
  - ~~~de-hackify switch between L2/tanh and cross-ent/sigmoid (KB) : N.B. I've done this but I can't test it because damn librosa broke somehow when my new machines
  were setup - scipy~~~
  - function naming convention is horrible; why do we have a `Pattern_Generator` module and `Feeder.Pattern_Generate`?
  - replace reading `.json` file with a `python` dictionary (possibly via attrs module) (KB, in progress)
  - ~~~convert `Pattern_Generator` from a script to a proper class, using the `PatternParameters` object and not a .json file (KB, in progress)~~~
  - do we need all these explicit conversions to float32??
  - patterns.PatternGenerator.generate_metadata() needs to be rethought
  - ~~~original Heejo code lists two words as "unrelated" if they are not identical, in the same cohort, or rhymes but THEN checks for 1-step neighborhood.  How can a word be called both "unrelated" and a DAS neighbor?? (If we start looking at neighbors, this is going to be super confusing.)~~~
  - ~~~need to write tests for phonology.py (though I know my edit_dist function works)~~~
  - need to write tests for new analyzer.py
  - ~~~need to add padding to the inputs in patterns.py~~~
  - we need to write noam weight decay as a custom learning rate schedule (has to be a LearningRateScheduler and passed in the callbacks to fit), or switch to a simpler weight decay
  scheme that keras already supports

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
