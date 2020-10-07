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
  - ~~make KB owner on `github` (JM)~~
  - convert to `python` module
  - better/standardized formatting (CB)
  - ~~integrate `github` with Slack~~
  - `excluded_Identifier`: can you useit to leave out a talker AND/OR a word (e.g. leave out all of Ava's items, or all instances of YELP, or AVA_YELP)?
  - add docstrings to methods/classes
  - `sphinx` for automatic documentation generation (CB)
  - add tests (via `pytest`): need to do testing on (1) Pattern_Generator, (2) Model, (3) Analyzer (KB will try to do this)
  - add everyone to R drive or lab NAS for big file storage (JM)
  - set long-term goals, e.g. commenting, keras-izing (ALL)
  - items for Gaskell/Marslen-Wilson semantic blends (JM)
  - de-hackify switch between L2/tanh and cross-ent/sigmoid (KB) : N.B. I've done this but I can't test it because damn librosa broke somehow when my new machines
  were setup - scipy
  - ~~~separate .json files into model, parameters, and analyzer sets (KB in progress)~~~
  - function naming convention is horrible; why do we have a `Pattern_Generator` module and `Feeder.Pattern_Generate`?
  - replace reading `.json` file with a `python` dictionary (possibly via attrs module) (KB, in progress)
  - convert `Pattern_Generator` from a script to a proper class, using the `PatternParameters` object and not a .json file (KB, in progress)
  - do we need all these explicit conversions to float32??
  - patterns.PatternGenerator.generate_metadata() needs to be rethought
  - ~~~need to catch exceptions in the ThreadPool used in patterns.PatternGenerator.generate_patterns() (KB)~~~

## Under Discussion:

(These items should either be moved into Tasks or deleted)

  - more flexibility in input patterns; have the model auto-sniff input/output sizes from a pattern file
  - shift to keras to clean up model instantiation etc.

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
