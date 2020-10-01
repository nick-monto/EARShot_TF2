# EARSHOT TODO

## People (for purposes of putative task assignment):

  - JM (Jim Magnuson)
  - KB (Kevin Brown)
  - CB (Christian Broadbeck)
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
  - separate .json files into model, parameters, and analyzer sets (KB in progress)

## Under Discussion:

(These items should either be moved into Tasks or deleted)

  - replace reading `.json` file with a `python` dictionary (possibly via attrs module)
  - more flexibility in input patterns; have the model auto-sniff input/output sizes from a pattern file
  - shift to keras to clean up model instantiation etc.
