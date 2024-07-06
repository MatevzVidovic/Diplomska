
- Kako oni režejo input channels:
Potem _prune_next_layer tem pobriše ta imnput channel. Ampak a ne bi to moralo imet potem še chain effecta za vse naprej?
- Nastavljivi parametri - kateri so zdaj, kaj vse bi še lahko
- Pogledat poslane modele
- Mogoče bi namesto IPAD lahko ocenili pomembnost filtra glede na njegov vpliv na pomembne filtre na naslednjem nivoju. Pač neka utežena vsota IPAD in tega. Kot en backpropagation.
Ali pa recimo za vsak filter bi lahko poračunal kakšen ima v povprečju na train setu vpliv na zaključno masko - samo ta del networka ki gre od njega naprej poženeš. Ampak za to bi recimo lahko namesto prav narejenih layerjev v init metodi lahko imel podane z dictionaryjem, in forward metoda potem recimo gre po zaporedju v tem seznamu oziroma drevesu. In potem bi komot klical forward samo od nekega indexa naprej in bi za nek filter točno dobil ves njegov efekt.
To zadnje je prekompleksno.
Tisto zgoraj pa kar implementiraj kejr je dokaj uporabno.



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Kako bi za rgb naredil, da imamo mošnost imet 3d kernel.
DUDE TO JE ŽE. V Unetu ima conv filter takšen channel count kot input. Po celi globini inputa se uči kernel. Kernel size je inchannelsx3x3. To je ves point.
- Šele zdaj mi je prebilo.




Aja, huh. Moramo podat te all_layers_params al kaj je, da ob ponovni naložitvi modela vemo kakšen je bil prvič.
Ampak ne, ker to bi komot pač v en json shranu.



Ničesar ne rabi podajat, ker mi pogledamo kateri modules so Conv2d in na njih lepo dela.

Samo poda naj, kaj je default limit za posamezen conv layer pa mogoče bi literally lahko za vsak module, ki ga noče preveč rezat, podal procent.
In najprej pogledamo tja. Če ga ni tam, pogledamo če je conv2 in gremo na default. Sicer pač ne režemo.

Kako oni režejo input channels:
Tako da za vsak index layerja poveš na katere indexe naprej kaže. Potem pa narediš recursive list kjer se bo rezalo - in povsod se enak filter ix reže, tako da moraš imet vedno te ta skip connections prej zložene v input filters.
Tako da bi ubistvu moral za vsak filter_ix tudi povedat, kako se naprej mappajo.
Lahko bi naredil, da se poda lambda funkcija:
f(start_layer_ix, start_filter_ix, goal_layer_ix) -> goal_filter_ix
Ki je v trenutnem načinu zlaganja samo: return start_filter_ix.
Ampak je pa problem, ker se inchanels ob rezanju spreminjajo in ne moreš kar tako preprosto hardcodeati te funkcije. Veliko lažje ej, če samo tako zlagaš, da so skip connections vedno na začetku.

V train_with_pruning_unet.py okrog line 530 funkcija disable_filter:
Poreže trenuten filter.
Potem pa:
layer_index = _get_layer_index(name, model)
    # prune next bn if nedded
    _prune_next_bn_if_needed(layer_index, index, index, 1, device, model)

    # surgery on chained convolution layers
    next_conv_idx_list = _get_next_conv_id_list_recursive(layer_index, model)
    for next_conv_id in next_conv_idx_list:
        #print(next_conv_id)
        _prune_next_layer(next_conv_id, index, index, 1, device, model)
In potem _get_next... poda kaj so naslednji layerji trenutnemu - tu je hardcoded, morda bojo te connections morali bit podani, mogoče bi se pa celo programabilno dalo dobit - čeprav pač. to je v forward metodi narejeno in bo pomoje treba podajat ker ne moreš kar dobit.
Za Unet je recimo:
if layer_index == 1:
            next_conv_idx = [2, 16]
        elif layer_index == 3:
            next_conv_idx = [4, 14]
Potem _prune_next_layer tem pobriše ta imnput channel. Ampak a ne bi to moralo imet potem še chain effecta za vse naprej?
        



Naredit možnost da tudi po ne-njihovi metodi (ne_IPAD) računamo pomembnost posameznega layerja. IPAD naj se poda kot lambda.

Vključit validacijsko, kjer ni simple validation set ampak podaš lambda funkcijo ki ima odvisnost od parametrov. Hranimo zadnnih k modelov v spominu in na koncu shranimo le najbolj perspektivnega.

Rezanje:
Za konvolucijo literally nardiš novo konvolucijo in batch norm, kjer ji za uteži samo tistega ixa ne prekopiraš noter.
Mogoče bi bilo bolje kot na roke to pač delat z nekim deepcopy pa potem samo uteži spremenit.

Naj ne odnehamo dokler nismo presegli in željenih FLOPsov in uteži. Da sam nastaviš, kaj ti je pomembneje. Pa recimo pri modelih s strideom (pa ta atrous splayed out conv kernel) je lahko velika razlika med tema dvema količinama.

Ocene:
Flops  - monkey patching resource manager dokaj elegantno. (Morda bi celo stride lahko vključil noter, ker to dosti zmanjša.)
Weights - ta calculate resources. Verjetno bi tudi lahko monkey patchal tho. Ampak,tole je tudi ok ig. Pač saj je isto samo d anamesto da forward dela zate, ti to poračunaš.

Data augmentation lepo zrihtat.
