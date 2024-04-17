
utils, opt, models, dataset so classi ki smo jih mi napisali.
Utils: tu je ResourceManager, ki pove, koliko flopsov je potrebnih.
Opt: to parsea ukaz za treniranje.
Models: tu naredimo dict objektov modelov ki bi jih ravno deklarirali.

op.attrgetter(block_name)(model)
To je operator.attrgetter.
Operator je native class. Za operacije in osnovne zadeve, kot je getting atributa, ti vrne funkcijo, ki to izvaja.
f = operator.attrgetter('name'), the call f(b) returns b.name
f = operator.add    # ta pac ne rabi dodatnega argumenta
>>> f(3,4)   printa 7

Torej:
op.attrgetter(block_name)(model)
on the fly naredi funkcijo f = op.attrgetter(block_name) in takoj zatem kliče f(model)

train_with_pruning_combined.py je glavna stvar.
Glavne funkcije, ki sem jih gledal:
- _get_sorted_filter_importance_dict_for_model
- disable_filter
- remove_filters_and_retrain_model
- count_learnable_parameters_for_module
- load_student
- start_pruning
- main

funkcija disable_filter je where the magic happens.
Tu dobimo konvolucijski layer in kater filter dat stran, se naredi prazna kopija tega layerja z enim out_channel-om manj. In potem se nastavijo weighti kot je treba.
Pomoje se tudi tu v nadaljnjih channelih popravijo inchannels, nisem pa še prišel do tega, kje se to dela.


funkcija remove_filters_and_retrain_model je glavna. Izvede en pruning in en retaining. To stori v treh korakih (več opisano spodaj).

funkcija main samo naloži model in potem zažene start_pruning, ki samo zaganja remove_filters_and_retrain_model.
Uporablja ResourceManager(model).calculate_resources, da preveri, koliko FLOPSov ima trenutno model, in breake-a if rm.cur_flops < targeting_flops.

ResourceManager se ne zdi prav zapleten. Ubistvu je v trace njegova glavna logika.
Tudi nisem ziher, zakaj se uporablja, ker je kinda že implementirano s funkcijo count_number_of_learnable_parameters.
Aha, je celo bilo tako uporabljeno, pa je zakomentirano. Me res zanima, kaj je šlo narobe.
Funkcija count_number_of_learnable_parameters potem samo gre po model.named_modules() in sešteje njihove count_learnable_parameters_for_module (opisano spodaj).



REMOVE_FILTERS_AND_RETRAIN_MODEL delovanje:

1. DOBI AKTIVACIJE
get_mIoU_on_train_and_populate_activations
(če si pozabil, activations so to, kar konvolucijski filtri vrnejo pri nekem primeru/batchu - ta activation deviation del)




1.1 KATERIH LAYERJEV IN CONVOLUTIONAL BLOCKOV NE SMEMO REZAT
Preverimo, če smo na nekem layerju ali bloku porezali že preveč parametrov in jih zdaj ne smemo več, da bo ostal uporaben.

Layer je kinda isto kot module. To je samo poimenovanje tako podvojeno.
Layer je v nn že implementirana zadeva (kot je nn.ConvTranspose2d() ali nn.MaxPool2d(2)).
To je ubistvu tudi block, ampak je pač že implementiran v nn in je tako osnoven. 

Block je pa class, ki implementira nn.Module. (torej ima forward()).
V forwardu lahko kot objekte vzame druge bloke in jih požene (To so lahko kompleksni bloki ali pa layerji), ali pa celo sam kaj dela.

Convolutional block je block, ki implementira več konvolucij.

Naš model je tudi implementira nn.Module in je ubistvu že sam block.

Tu naj spomnim, da je nn.Sequential() tudi nn-jev class, ki je block. Basically njemu kot argumente podaš blocke/layerje in jih on zaporedno izvede. To je samo pač olajšava, da ti ne rabiš potem v forwardu vsega tega zaporedno klicat (+ verjetno je o zadaj potem v C-ju ali cudi napisano in je hitreje).



Mi filtriramo samo konvolucijske layerje. Tudi FLOPsi nas zanimajo samo zanje. Zagotovili bi radi, da ima vsaka konvolucija dovolj filtrov, da je sploh še učinkovita.
Zato bomo filtre layerjev, ki smo jim parametre posekali že pod nek procent originalnih parametrov (njihovo število najdemo v spodaj opisanih seznamih) izvzeli iz potencialnih filtrov za rezanje.

Imamo pa tdi konvolucijske bloke, ki imajo več zaporednih konvolucij. Tu tudi ne bi radi, da je v verigi kot celoti premalo parametrov. Zato tudi za te low level bloke izvedemo štetje parametrov in njihove filtre izvzamemo iz rezanja.


V start_pruning() mi za svoj model definiramo dva dictionaryja, naprimer:

    elif 'unet' in model_name:
        all_parameters_for_block_dict = {
            'inc': 37568,
            'down1.conv':  221440,
            'down2.conv':  885248,
        }
        all_parameters_layer_name_dict = {
            'down1.conv.conv1': 73856,
            'down1.conv.conv2': 147584,
            'down2.conv.conv1': 295168,
        }

Zaradi teh imen nam tudi lepo koristi:
layer_name.rpartition('.')
rpartition je native python funkcija:
Search for last occurrence of the string and return a 3-tuple:
(str before the "match", "match", str after "match")


funkciji get_curr_params_for_block in get_curr_params_for_layer basically samo uporabita count_learnable_parameters_for_module.

funkcija count_learnable_parameters_for_module
(preimenovana bi morala bit v: count_learnable_parameters_for_layer, ampak ime izhaja iz model.named_modules() ker je tako for some reason)
 uporabi funkcijo
count_zeroed_filter_for_layer.
Ta gre po layer filtrih in weights pa biases pa še nekaj primerja s torch.zeros(neki). Če je za vse enako (so zeroed), doda filter med zeroed, sicer med used.
count_learnable_parameters_for_module vzame len teh dveh listov in iz tega glede na velikost filtra preračuna število parametrov.




