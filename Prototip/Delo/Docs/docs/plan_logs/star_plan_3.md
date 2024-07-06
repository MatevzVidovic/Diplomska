Zdaj je vprašanje:
Js nardim tak class kot v UOZP nalogi in on mi da objekt modela na katerem kličem forward. Pa poda mi dict s parametri kot je learning rate in weight decay.
Anpak potem rabim za razne možnosti pokrivat opcije - pač vsako stvsr kot je weight decay moram imet kot opcijo, ampak ne morem imet vseh, ker bo čisto intractible sprogramirat za vse te parametre.

Ali mi pa on poda tak class, kjer je sam določil weight decay in to, pa ima svojo train in svojo test metodo, pa ima svoje dataloaderje.
Pač nardil bom, da za majhen nabor parametrov mi lahko samo objekt modela poda pa recimo weight decay pa learning rate pa take.
Če pa hoče bl kompleksno, pa čist simply sam nardi class po tem templateu, kot sem že jaz naredil - pač da ima train in test in train test loop kjwr se potem model shrani.
Vse ostalo pa on sam spiše kakor mu paše.


Problem: kako bo on sploh vedel kako naredit to filter_ix lambdo. Ker on ne ve kakšen je layer_ix v mojo arhitekturi.
Mogoče mi bo on moral zraven še podat preslikovalni seznam kakšne layer_ix imajo vsi njegovi zaporedni conv filtri.

Torej on mi poda:
- Objekt svojega modela,
- lambda funkcijo ki pove goal_filter_ix iz (curr_layer_ix, curr_filter_ix) (naj on pač zapiše ta pravila stackinga na roke z elifi. Ali pa pač poda kar seznam ki to narekuje in to da v to lambda funkcijo. Sej bolj elegantnega načina nimamo.),
- lambda za minimum delež ohranjenih flopsov (in uteži, in filtrov) v odvisnosti od layer_ix (če oseba hoče, lahko sama poda natančen list v to funkcijo, in ta lambda potem samo bere od tam. Doesnt matter.
Samo tako omogočim, da oseba ne rabi nujno podajat seznama, ampak lahko recimo linearno funkcijo samo poda - pač tok ekstra easy je neko funkcijo vsem skupej dat, če hočeš. Ta design peoncip mi je res všeč, ker pač tok improvea to, da bi blo nujno treba list dodajat.


Jaz potem glede na model in lambda funkcijo zgradom dictionary:
LayerIx : list kjer za vsakega od trenutnih neporezanih filtrov kakšen je rezultat lambda funkcije.
In to naredim na začetku, potem pa vsakič ko režem uteži, zraven se porežem tole.

Potem samo naredim ConvResourceCalc in preverim, če so kateri že pod mejami.
!!!! Uporabnik na štartu poda: float za procent min ohranjanja v temle layerju. Al pa poda lambda ki iz layer ix pove kakšna je zanj meja. In če je tu None ali pa tudi 0.0 ker potem defecto ni meje, potem preverimo, če je treba ta layerix odstranit iz circulationa.

Potem dobimo trenutne resources in sortiramo filtre in vzamwmo prvega, ki ni med temi, ki ne smejo bit več. In tega porežemo, in porežemo filtre na katere se povezuje iz tega podanega dictionaryja.




Naredit osnovni trainer ki je mešanica med demo pa tem pristopom, ki sem ga uporabil za UOZP nalogo.

Pa ta trenutni retrainer naredit boljši.

Ne pozabit upoštevat strides in attrition in padding. - heh ubistvu nič ne rabim 
A nam RM da za celotwn network naenkrat?
A bo nam lahko dictionary kjer layerjevo ime ali pa index kaže na njegovo število floppov oziroma uteži oziroma število filtrov.9ppb


vzet RM in še monkey patch za število conv uteži.
Preimenovat ga v ConvResourceManager

Pri rezanju naj za nov filter naredim copy in potem te weights spremenim.
Ker zdaj je ogromno ene kode ki nima kaj delat tu.

Njegov dataloader vzet, oziroma kar tega, ki ga imam zdaj v tem svojem demotu.
