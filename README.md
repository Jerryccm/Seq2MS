# Seq2MS
Seq2MS



## How to use

__Before running, pretrained model `long1024_simple_extend_30epoch` can be downloaded from [google drive] and should be placed into this directory.__

### Notes

* Modifications should follow the format in Maxquant outputs 
* The prediction will NOT output peaks with M/z > 2000
* Minimum intensity for predicted peaks to be present in output library is 0.5% of the strongest peak in spectrum

### Required Packages

* Python >= 3.7
* Tensorflow >= 2.3.0
* Pandas >= 0.20
* pyteomics
* lxml

### Input format

The required input format is .tsv, with the following format (Column order can be flexible):

Sequence | Charge | Mass | Modified sequence | Modification | Protein
------- | ------ | ---- | ----------------- | ------------ | -------
AAAAAAAAAAAAAAAGAGAGAK | 2 | 1597.8525 | AAAAAAAAAAAAAAAGAGAGAK 
AAAAADLANR | 2 | 944.5029 | AAAAADLANR |  
AAAAGSLDR | 2 | 832.4392 | AAAAGSLDR | 

More examples are available in `example.tsv`.

### Usage

For predicting spectral library from list of peptide inputs, run:

`python predict.py --input example.tsv --model long1024_simple_extend_30epoch --output predicted_library.msp`

The output file is in .msp format

* --input: the input file
* --output: the output file , (optional, output filename will default to the same as input name)
* --model: the pretrained model

For evaluating prediction accuracy with existing library, run:

`python evaluate_model.py --data hcd_testingset.mgf --model long1024_simple_extend_30epoch`

* --data: the input library
* --model: the model being evaluated

To train your own model from scratch, run:

`python train_model.py`

The default data source ProteomeTools.mgf is available for download at.
Users will need to edit manually if they wish to use different training data
