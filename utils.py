from pyteomics import mgf
import numpy as np
import tensorflow as tf
from tensorflow import keras as k

SPECTRA_DIMENSION = 20000
BIN_SIZE = 0.1
MAX_PEPTIDE_LENGTH = 50
MAX_MZ = 2000
batch_size=32

def parse_spectra(sps):
    db = []

    for sp in sps:
        param = sp['params']

        c = int(str(param['charge'][0])[0])
        if 'seq' in param:
            pep = param['seq']
        else:
            pep = param['title']

        leave = False
        for i in range(len(pep)):
          if pep[i] not in charMap.keys():
            leave = True
        if leave == True:
#           print('have mod, skipped')
          continue

        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])
        mz = sp['m/z array']
        it = sp['intensity array']

        db.append({'Sequence': pep, 'Charge': c,'Modified sequence':pep,'Modifications':'',
                   'Mass': mass, 'mz': mz, 'it': it, 'len':len(pep)})

    return db


def readmgf(fn):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False,
                    dtype='float32', use_index=False)

    codes = parse_spectra(data)
    return codes
    
Alist = list('ACDEFGHIKLMNPQRSTVWY*')
encoding_dimension = 12

charMap = {}
for i, a in enumerate(Alist):
    charMap[a] = i + 1
    

types = (tf.float32, tf.float16)
shapes = ((27, encoding_dimension), (20000))


def find_mod(row):
    row = row.strip('_')
    seq = row#[1:-1]
    pos = 0
    poslist = []
    modlist = []
    ismod = False
    mod = ""
    for i in range(len(seq)):
       
        if seq[i] == ')':
            ismod = False
            pos -= 1
            poslist.append(pos)
            modlist.append(mod)
            mod = ""
        if ismod == True:
            mod += seq[i]
        else:
            pos += 1
#             print(pos)
        if seq[i] == '(':
            ismod = True
            pos -= 1
    return modlist, poslist
 
mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538}

ave_mass = {"A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 160.1598, "E": 129.1155,
            "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594, "L": 113.1594, "K": 128.1741,
            "M": 131.1926, "F": 147.1766, "P": 97.1167, "S": 87.0782, "T": 101.1051,
            "W": 186.2132, "Y": 163.1760, "V": 99.1326}
# Amino acid : [C H N O S P]
atoms = { "A": [1,2,0,0,0,0], "R":[4,9,3,0,0,0], "N":[2,3,1,1,0,0], "D":[2,2,0,2,0,0], "C":[1,2,0,0,1,0], "Q":[3,5,1,1,0,0], "E":[3,4,0,2,0,0], "G":[0,0,0,0,0,0],
          "H": [4,4,2,0,0,0], "I":[4,8,0,0,0,0], "L":[4,8,0,0,0,0], "K":[4,9,1,0,0,0], "M":[3,6,0,0,1,0], "F":[7,6,0,0,0,0], "P":[3,4,0,0,0,0], "S":[1,2,0,1,0,0],
          "T": [2,4,0,1,0,0], "W":[9,7,1,0,0,0], "Y":[7,6,0,1,0,0], "V":[3,6,0,0,0,0] 
}

# C H N O S P
mods = {"ox" : [0,0,0,1,0,0], 
        "ph" : [0,1,0,3,0,1], 
        "cam" : [2,3,1,1,0,0] , "ac": [2,2,0,1,0,0], "me": [1,2,0,0,0,0], "hy": [0,0,0,1,0,0], "gly": [4,6,2,2,0,0],
        "bi" : [10,14,2,2,1,0], "cr": [4,4,0,1,0,0], "di": [2,4,0,0,0,0], "ma": [3,2,0,3,0,0], "ni": [0,-1,1,2,0,0],
        "bu" : [4,6,0,1,0,0], "fo": [1,0,0,1,0,0], "glu": [5,6,0,3,0,0], "hyb": [4,6,0,2,0,0], "pr": [3,4,0,1,0,0],
        "su" : [4,4,0,3,0,0], "tr": [3,6,0,0,0,0], "ci": [0,-1,-1,1,0,0]}
        
mass_weight = np.array([0.06,0.005,0.07,0.08,0.16,0.155])

def asnp(x): return np.asarray(x)
def asnp32(x): return np.asarray(x, dtype='float16')

def spectrum2vector(mz_list, it_list, bin_size, charge):
    
    it_list = it_list / np.max(it_list)

    vector = np.zeros(SPECTRA_DIMENSION, dtype='float32')

    mz_list = np.asarray(mz_list)

    indexes = np.floor(mz_list / bin_size)
    indexes = np.around(indexes).astype('int32')
    indexes = np.clip(indexes,0,SPECTRA_DIMENSION-1)

    for i, index in enumerate(indexes):
        vector[index] += it_list[i]

    # if normalize
#    vector = np.sqrt(vector)

    return vector

def embed(sp, charge, mass_scale=MAX_MZ):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float32')

    pep = sp
    for i in range(len(pep)):
        if i >= MAX_PEPTIDE_LENGTH:
          break
        encoding[i][:6] = atoms[pep[i]] * mass_weight

    encoding[-1][charge] = 1

    return encoding    

def embed_maxquant(sp, mass_scale=MAX_MZ, augment=True, fixedaugment = False, key=None, havelong=False):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float16')

    pep = sp['Sequence']
    charge = int(sp['Charge'])
    if augment:
        pos_to_mod = np.random.randint(0, len(pep))
        roll = np.random.uniform(0,1)
    for i in range(len(pep)):
        if fixedaugment and pep[i] == key:
          encoding[i][6:12] = atoms[pep[i]] * mass_weight
          continue
        if i >= MAX_PEPTIDE_LENGTH:
          encoding[-2][:6] += atoms[pep[i]] * mass_weight
          continue
        if augment and roll <= 0.1:
          if i == pos_to_mod:
            encoding[i][6:12] = atoms[pep[i]] * mass_weight
            continue
        encoding[i][:6] = atoms[pep[i]] * mass_weight
    encoding[-1][charge] = 1    
    encoding[-1][-1] = sp['NCE'] / 100 if 'NCE' in sp else 0.25    

    #add modification
    modlist, poslist = find_mod(sp['Modified sequence'])
    for i in range(len(poslist)):
      if modlist[i] == "gl":
        modstring = sp['Modifications'].split(',')
        if "Glutaryl [K]" in modstring:
          modlist[i] = "glu"
        else:
          modlist[i] = "gly"
      if modlist[i] == "hy":
        modstring = sp['Modifications'].split(',')
        if "Hydroxyisobutyryl [K]" in modstring:
          modlist[i] = "hyb"
        else:
          modlist[i] = "ox"
      #encoding[poslist[i]][:6] += mods[modlist[i]] * mass_weight # prevent double count if modified glycine
      encoding[poslist[i]][6:12] += mods[modlist[i]] * mass_weight
    return encoding

def embed_pdeep(sp, mass_scale=MAX_MZ):
    encoding = np.zeros((MAX_PEPTIDE_LENGTH + 2, encoding_dimension), dtype='float16')

    pep = sp['peptide']
    charge = int(sp['charge'])
    for i in range(len(pep)):
        if i >= MAX_PEPTIDE_LENGTH:
          break
        encoding[i][:6] = atoms[pep[i]] * mass_weight
    encoding[-1][charge] = 1       

    #add modification
    allmods = sp['modification'].split(';')
    for i in allmods:
        modstring = i.split(',')[1][:2]
        modstring = modstring.lower()
        pos = int(i.split(',')[0])
        if modstring == "gl":
          modstring = i.split(',')[1][:2].lower()
        if modstring == "hy":
          modstring = i.split(',')[1][:2].lower()
        encoding[pos][:6] += mods[modstring] * mass_weight
        encoding[pos][6:12] += mods[modstring] * mass_weight
    return encoding
    
def spectrum2vector_maxquant(mz_list, it_list, bin_size, charge):
    it_list = it_list.strip('[]\n\t')
    mz_list = mz_list.strip('[]\n\t')
    it_list =  [float(idx) for idx in it_list.split(';')]
    mz_list = [float(idx) for idx in mz_list.split(';')]
    it_list = it_list / np.max(it_list)

    vector = np.zeros(SPECTRA_DIMENSION, dtype='float16')

    mz_list = np.asarray(mz_list)
    mz_list[mz_list>1999.8] = 1999.8

    indexes = np.floor(mz_list / bin_size)
    indexes = np.around(indexes).astype('int16')

    for i, index in enumerate(indexes):
        vector[index] += it_list[i]

    # if normalize
    #vector = np.sqrt(vector)

    return vector

def spectral_angle(true, pred):
    cos_sim =  tf.keras.losses.CosineSimilarity(axis=1)
    
    product = cos_sim(true,pred)
    arccos = tf.math.acos(product)
    return 2 * arccos / np.pi
    
def masked_spectral_distance(true, pred):
    epsilon = k.backend.epsilon()
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    pred_norm = tf.math.l2_normalize(true_masked, axis=-1)
    true_norm = tf.math.l2_normalize(pred_masked, axis=-1)
    product = k.backend.sum(pred_norm * true_norm, axis=1)
    arccos = tf.math.acos(product)
    return 2 * arccos / np.pi

def write_msp(out, sps, peps):

    def f2(x): return "{0:.2f}".format(x)

    def f4(x): return "{0:.4f}".format(x)

    def sparse(x, y, th=0.02): #0.02
        x = np.asarray(x, dtype='float32')
        y = np.asarray(y, dtype='float32')
        y /= np.max(y)
        return x[y > th], y[y > th]

    def write_one(f, sp, pep):
        precision = 0.1
        low = 0
        dim = 20000
        #pep['Mass'] = 1000
        #sp[min(math.ceil(float(pep['Mass']) * int(pep['Charge']) / precision), len(sp)):] = 0
        imz = np.arange(0, dim, dtype='int32') * precision + low  # more acurate
        mzs, its = sparse(imz, sp)
        
        seq = pep['Sequence']
        charge = pep['Charge']
        mass = pep['Mass']
        protein = pep['Protein']
        modlist, poslist = find_mod(pep['Modified sequence'])
        for i in range(len(poslist)):
            for mod in pep['Modification'].split(','):
                if mod[:2].lower() == modlist[i]:
                    modlist[i] = mod.split(' ')[0]
                    break

        modstr = str(len(poslist)) + ''.join([f'({p},{seq[p]},{m})' for m, p in zip(modlist, poslist)]) if poslist != [] else '0'
        head = (f"Name: {seq}/{charge}_{modstr}\n"
                f"Comment: Charge={charge} Parent={mass/charge} Mods={modstr} Protein={protein}\n"
                f"Num peaks: {len(mzs)}\n")
        peaks = [f"{f2(mz)}\t{f4(it * 1000)}" for mz, it in zip(mzs, its)]

        f.write(head + '\n'.join(peaks) + '\n\n')

    for i in range(len(peps)):
      write_one(out, sps[i], peps.iloc[i])
