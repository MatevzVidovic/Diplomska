
- da model wrapperju lahko podaš dataloaders, optimizer, loss fn. Lahko pa podaš parametre zanje, in ti jih mi ustvarimo.
Še bolje je pa, da sam napišeš ta wrapper.
Moj algoritem deluje tako - ti ustvariš objekt takšnega wrapperja, ki sem ga napisal (ima metode train, test, save - to je vse, kar bom jaz uporabljal).
Pa z mojim wrapperjem si lahko spomagaš, da napišeš svojega. Lahko pa svoj wrapper object ustvariš prek tega, kar sem napisal.

- Naredit, da se ta splošen disctionary naredi posebej nekje drugje. Samo resources dict se dela v ConvCalc
- naredit ConvCalc bolj pregledno skodiran.
- v ConvCalc da flops upoštevajo stride (čeprav ker gledamo output velikost je to že noter) (in morda atrious al kaj so - sam kaj bi sploh to pomenil tle)
- kako naredit da za conv calculstor ni treba podajat pravilne velikosti vnos? Pač da bi vedeli input dimensions za network oziroma prvega layerja in ne bi bilo treba ročno ustvarit tega tensor(1, 1, 128  128)

- je ta ideja s tem dictom dobra? Pomoje je. Ke rto uporabnik uporablja samo za meje rezanja in za skip connections - in pri obojih ponavadi je nek vzorec v številkah, glede na katerega lahko programatically napišejo, kako nastaviti ta dict. Če pa se ne da programatically pa itak morajo na roke, ker boljše opcije ni. Mogoče kakšen list ustvarit pa to po vrsti filat noter ali kaj takega.
Mogoče edino naredit opozorilo, če so nastavili mejo za module, ki v sebi nima nobene konvolusije - da opozori na možno napako. Pa naj se za neničelne meje naredi izpis ob zagonu kateri (tree_ix, layer_ime, nastavkjena meja) so neničelni.
In pomoje je kul, da imaš možnost za nek module po imenu podat mejo rezanja, in tudi za vsakega specifično, kar pa tudi overridea po imenu podano specifikacijo. (Po imenu podana lahko komot preverjamo s tem, da je podana v obliki, kot je v tem models to names dict, ali pa če samo tree ix od enega primerka tega kar hočeš podaš tudi lahko ugotovimo).

- pomoje se splača uporabniku zgradit list ki ima po vrsti tree_ix od vseh conv layerjev po vrsti. Tako bo imel shorthand alla conv[2] kar pomeni tretja konvolucija. Pač kdaj zna pomagat.
OZIROMA VELIKO BOLJE: funkcija, kjer uporabnik poda ime layerja, in mu mi damo list, kjer so zaporedno tree_ix za te layerje, kot se pojavijo v mreži. Tako bo za vse, čemur hoče nastavljat posebne meje rezanja, pa hoče nastavljat skip connections, imel priročen in easily razumljiv shorthand.
Bolj simple kot to se preprosto ne da naredit.


- lambda ki povezuje filtre za skip connections. F(tree_ix, filter_ix)->list of (goal tree ix, filter_ix). Tako vemo, na katere vse vpliva.
Če je potrebno lahko tudi potem to rekurzivno naprej izvajamo - ampak kakor razumem se to trenutno ne dogaja

- kako so izvajali oni training? Literally na roke počeli to, kar jaz delam s train()?
- kaj je ta LeGR codebase
- kako tf so conv2d tu notri, če naj bi delal na vseh channelih/filtrih naenkrwt? Pač če je 2d, po čem sploh gre? Gre vsak filter samo po soležnem filtru? In so soležni filtri ubistvu torej samostojen chain do konca mreže?


