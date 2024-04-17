
- s sem jaz v DiplomskaDemo pozabil:
model.to_device() in je bilo potem ekstra počasi pa sploh ne bi rablo bit?

- overview kakšen je torej pipeline - kaj mi morajo podat in kako splošno lahko naredim tole.

- kaj so student in teacher modeli, in zakaj so posebej definirani v models.py
Pač kaj je s tem.

- še 2. step remove_filters_and... je za raziskat - kako se izračunajo aktivacije inkako se med njimi potem izbere filtre za brisat (upoštevajoč kere filtre ne smemo brisat)

- 3. step od remove_filters_and... trivialen za napisat

Kaj the fuck naj bi blo tole:
utils, resource manager
    def _is_leaf(self, model):
        return self._get_num_gen(model.children()) == 0

model.named_modules()
model.named_parameters()

kaj so vse te zadeve? Od kje pridejo? Ker v original classih jih ni. A to jih mi kaj obogatimo, ko jih ustvarimo, al od kje je kar to?
