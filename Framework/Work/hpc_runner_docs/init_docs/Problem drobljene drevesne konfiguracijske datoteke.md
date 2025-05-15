99999999x!!!!!!! Problem drobljene drevesne konfiguracijske datoteke

keywords: sbatch, hpc, frida, runner

Za hoc running in zabunrestricted running bi bilo dobro imet arg ki je flag, a upprabi bash ali sbatch za ukaz.
Tako lahko delaš runner.py in ima on svoje ene frida argse in potem runna te filee



Root dir
 ali z fileom prisotnost katerega oznacuje to in potem greß iz file-pwd proti parentu in ko naletiß nanj veß, kateri je
Ali pa zaganjanje iz njega in je kot pwd ukaz

Sbatch meja:
Frida runner + sequential runner (kot zdaj bash). Popen uporablja. Je za mejo sbatcha, torej omejen v enem vozlišču, in ne more več klicsti sbatcha in tako vzporediti. Lahko pa še vedno sh vzporedi (kliče več bash ukazov s popen in jih nato šele počaka). Ampak pač vram problems ane.
Nonconstrained runner: še pred sbatch mejo se izvaja. Kliče vse ukaze z sbatch in uporablja popen in tako lahko vzporedi.


Od roota pa do runnerja se nabirajo dirdef.yaml files.
Te imajo args za frida running,
pa args za bash running (kot je bilo sedaj te constants).
V mapi tega run-pythona je tudi še en yaml lahko, ki ima isto ime kot ta run-python, in ta se še potem overdrivea.



Yaml_id in yaml override:
Kako specifyjqt?
Z le imenom (v katero mapo potem gleda?)
S pathom od roota do yaml filea - to je kul, ampak dajmo še omogočit relative paths, torej npr ./tule.yaml

Oba ta pristopa bi želel imet hkrati.

Z le imenom rešiš tako:
- yaml/ po poti ideja
- PATH ideja

Yaml po poti:
Od roota do trenutnega imaš lahko v tej mapi še yamls mapo.
Če daš samo ime, bo pogledal najprej v trenutna/yamls/, potem v (../trenutna)/yamls/


PATH pristop za yaml podajanje
Od roota do . Imaš dirdef.yaml
V njem lahko podaš yaml_dir_paths
To je list tupleov v yamlu
Tuples izgledajo tako:
(Crit level, yaml_dir_path)
Yaml dir path je lahko abs path od roota, ali pa je relative path glede na to kje smo trenutno z dirdef.yaml

Cdit level je lahko 0-9999
Imamo pa še 2 lista pathov : te pred crit levels in te za crit levels.
-1 je, da pomembnejši od vseh
-2 da se appenda listu teh, ki so pred crit levels
-3 prepend tem po ceit levels
-4 append tem po crit levels
In to imaš potem res absolutno svobodo.

In recimo po avtomatskem se ./yaml da z -1 v to
(Tako združimo z "yaml po poti" idejo)
Imaš pa potem arg, ki je recimo:
Yaml po poti: False
In se tako to ne zgodi




Kako pa podajamo argparse od bash-py-ja?
Pač to kar smo prej kot arge podajali?
Ja pač ponovno kot args.
Frida runner bo moral sprejet en dict arhumentov, ki va bo potem podal naprej.
Ali pa recimo en string argumentov - kot bi vse args podal v narekovajih).
In rdcimo da tu lahko podamo katerikoli frida arg, katerikoli bash arg, katerikoli yaml file arge (samo ime in je isti postopek kot zgoraj, ali pa damo abspath, ali pa damo relpath kjer je ./ tam kjer se bash-run-py nahaja
Lahko vi celo naredil, da lahko podaš katerikoli yaml arg, torej to, kar se znotraj yamlov sicer podaja, in se se tako deeply lahko overridea.


Razlika med frida run in pa unconstrained je
Da frida run pobere dirdef args po poti, potem pa vzame frida args, in požene z sbatch bashpy in mu poda ostale argumente kot args.

Unconstrained pa sbatchpy-u poda vse args in on potem uporablja te frida args.
Pa frida args so lahko dict
In potem ima lahko za vsak zagon sbatcha znotraj sbatchpy-a svoje frida args, kar je tudi smiselno.




