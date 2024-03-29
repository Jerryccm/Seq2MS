# Seq2MS
Seq2MS is a neural network tool to predict tandem mass spectrum for given peptide sequences.

DOI: https://doi.org/10.1021/acs.jproteome.3c00180

![](https://github.com/Jerryccm/Seq2MS/blob/master/nist_common_3plot.svg)

## How to use

__Before running, pretrained model `seq2ms_pretrained_model.zip` can be downloaded from [google drive](https://drive.google.com/drive/folders/16gbW6qa2KdBkDOvyG6McpyUjs-I9jaRc?usp=drive_link) or ProteomeXchange Consortium via the dataset identifier PXD040721 and should be extracted and placed into this directory.__

## Resources
Prebuilt spectral libraries available for download from [google drive](https://drive.google.com/drive/folders/16gbW6qa2KdBkDOvyG6McpyUjs-I9jaRc?usp=drive_link)

### Notes

* Modifications should follow the format in Maxquant outputs 
* The prediction will NOT output peaks with M/z > 2000
* Minimum intensity for predicted peaks to be present in output library is 0.5% of the most intense peak in spectrum

### Required Packages

* Python >= 3.7
* Tensorflow >= 2.3.0 <= 2.8
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

`python predict.py --input example.tsv --model pretrained_model --output predicted_library.msp`

The output file is in .msp format

* --input: the input file (.tsv)
* --output: the output file , (optional, output filename will default to the same as input name)
* --model: the pretrained model name
* --mod: (Optional, str ) For defining custom modifications in the form of a dictionary: "{'custom':[9,9,9,9,9,9]}" # Atom counts [C,H,N,O,S,P]

For evaluating prediction accuracy with existing library, run:

`python evaluate_model.py --data hcd_testingset.mgf --model pretrained_model`

* --data: the input library
* --model: the model being evaluated

To train your own model from scratch, run:

`python train_model.py`

The default sample data source `ProteomeTools.mgf` is available for download at PXD040721.
Users will need to edit manually if they wish to use different training data

## Example workflow

Download Uniprot Protein database fasta file:

![image](https://github.com/Jerryccm/Seq2MS/assets/53712272/a6733114-e99e-4f1e-a65d-324905e2a036)

Digest fasta file to obtain peptide list:

![image](https://github.com/Jerryccm/Seq2MS/assets/53712272/05698e32-43b3-4210-bcb4-0ac39c2e11be)

Convert peptide list (.txt file) into input format with digested_to_input.py (default predicts all peptides for charge states 2 and 3, modify for higher charge states):

`python digested_to_input.py <peptide.txt> <input.pkl or input.tsv>`

Then run predict.py with the input:

`python predict.py --input input.tsv --model pretrained_model --output predicted_library.msp`

Importing msp library to spectrast: 

`spectrast -cN<library name> predicted_library.msp`
if it is a decoy library, then:
`spectrast -cmDECOY -c_NPK0 -cN<library name> predicted_library.msp`
