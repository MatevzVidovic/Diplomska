
### Skip connections in pa croppanje v Unet članku, ki ga pri nas ni:
V Unet članku v shemi arhitekture nam črtkana črrta na zadnjem od treh layer-jev vsakega downsampling-a govori o cropp-anju, ki je pomemben za skip connections. V članku uporabljajo konvolucioj brez mirroring-a robov, zato izgubljajo rob z vsako konvolucijo. Piksli zunaj črtkane črte pomoje sploh ne pridejo do zadnjega downsampling-a (vsak naslednji layer prvotni sliki odvzame eksponentno več robnih pikslov, saj v sliki v kasnejšem downsamplingu vsak piksel predstavlja 2x2 pikslov predhodnega layer-ja.

Naša implementacija Unet-a uporablja mirroring, zato do tega croppanja ne pride. Poleg tega za velikosti uporablja potence števila 2, zato se ob downsamplanju ne izgubi noben dodaten piksel. 

Iz tega razloga so skip connections še posebej preprosti - v UNet v forward metodi v upsampling blok le dodamo ves izhod corresponding downsampling bloka.
Ker za velikosti uporabljamo potence števila 2 in se piksli ne izgubljajo, sta v Up.forward() diffX in diffY vedno 0 in ne potrebujemo padding-a.


### Data Augmentation če le enkrat inicializiramo dataloader:
Mislil sem, da ker pri treniranju med epochi ne ustvarjam novega dataloadinga, ni data augmentationa in so zato samo iste slike, isto popačene kot ko so prvič naložene.
Ampak se zdi, da to ni res.
Pri test_purposes=True je vse ekstra hitro ker se slike z __get_item__ nalagajo šele med učenjem. Torej DataLoader vsakič kliče __get_item_, kjer pa se izvaja tudi data augmentation.


### Zanimivost z DataLoader:
Opazil sem ponavljanje path-a slike, zato sem šel preverjati, zakaj se je to zgodilo in če je kje kakšen bug - tega sem se lotil v testiranje_pravilnosti_skip_connectionov.
V IrisDataset sem v init-u naredil prazen list self.paths_so_far, in v __get_item__ sem append-al v ta list, da bi preverjal, če se je kakšen image_path že pojavil. Med zaporednimi ponovnimi izvajanji (ko se je model shranil in je program začel while loop znova) sem imel problem, da se je ta list ponovno izpraznil in mi ni bilo jasno, kaj se dogaja.
Ta list sem prestavil iz razreda med statične spremenljivke programa, a to ni pomagalo.
Za dataloader sem uporabljal num_workers=4 in num_workers=1. Zdi se, da te worker-ji naredijo svojo kopijo dataset razreda. Zato ob ponovnem ciklu dobimo ponovno kopiranje razreda, in je self.paths_so_far prazen.
Ko sem uporabil num_workers=0, ki loading izvaja v main procesu, so se začeli path-i ponavljati. Na neki točki ni bilo več novih path-ov.
(Del problema je bil tudi, da sem na začetku programa vzel prvi batch, da sem preveril njegove dimenzije, in je zato kasneje prišlo do ponavljanja.
https://discuss.pytorch.org/t/dataloader-resets-dataset-state/27960/3





### Opozorilo:

Pojavi se razlika v F1 in IoU, ko sta izračunana za tenzor batch-a in ne kot povprečje izračuna za vsako matriko v batch-u posebej. Verjetno ker pač rounding errors.:
Test Error: 
 Avg loss: 0.06256591 
 IoU: 0.978237 
 F1: 0.924424 
IoU_as_avg_on_matrixes: 0.978292
-------------------------------
Test Error: 
 Avg loss: 0.06256591 
 IoU: 0.978246 
 F1: 0.924281 
IoU_as_avg_on_matrixes: 0.978292

