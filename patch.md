VELA — Patch Note conceptuelle
Objet du patch

Cette note formalise une évolution importante de la définition de VELA.

La version précédente de VELA-Belief formulait principalement le problème comme l’apprentissage d’un état latent compact et persistant :

[
x_t = E_\theta(o_t),
\qquad
z_t = U_\theta(z_{t-1},x_t),
]

avec l’idée future de remplacer le vecteur latent unique (z_t) par une mémoire composée de plusieurs slots :

[
M_t\in\mathbb R^{K\times d_m}.
]

La nouvelle formulation précise beaucoup plus clairement :

la structure de cette mémoire ;
la manière dont elle est mise à jour ;
le rôle de l’attention dans l’allocation de mémoire ;
la notion de consolidation et de fusion de concepts ;
les pressions architecturales nécessaires à l’émergence d’une mémoire structurée ;
la distinction entre mémoire VELA et contexte actif d’un Transformer ;
l’utilisation de VELA comme mémoire long terme lors de la saturation du contexte d’un LLM ;
l’utilisation de la même architecture comme hidden state structuré d’un world model.

L’objectif de VELA n’est donc plus seulement d’apprendre une représentation compacte du passé.

Il devient :

Apprendre une mémoire latente persistante, bornée, structurée et auto-organisée qui conserve, met à jour, consolide et oublie l’information en fonction de sa valeur prédictive future.

1. Nouvelle représentation de la mémoire

Au lieu d’un unique vecteur latent :

[
z_t\in\mathbb R^d,
]

VELA utilise une mémoire structurée :

[
\boxed{
M_t=
\begin{bmatrix}
m_t^1\
m_t^2\
\vdots\
m_t^K
\end{bmatrix}
\in\mathbb R^{K\times d_m}
}
]

où :

(K) est le nombre maximal de slots mémoire ;
(d_m) est la dimension d’un slot ;
(m_t^k) représente une unité de mémoire persistante.

La taille de la mémoire reste donc bornée :

[
|M_t| = Kd_m
]

indépendamment de la longueur totale de l’historique.

La mémoire n’est pas conçue comme un simple ensemble des (K) derniers tokens ou des (K) tokens ayant reçu le plus d’attention.

Chaque slot est destiné à devenir une représentation latente persistante d’une information ou d’un facteur du monde.

Exemples possibles :

[
m_1 \sim \text{état d’un objet},
]

[
m_2 \sim \text{objectif courant},
]

[
m_3 \sim \text{information sur un agent},
]

[
m_4 \sim \text{régime dynamique},
]

[
m_5 \sim \text{relation persistante entre plusieurs entités}.
]

Ces interprétations ne sont pas imposées manuellement.

Elles doivent émerger de l’optimisation.

2. Distinction fondamentale entre token memory et semantic memory

VELA ne cherche pas simplement à réduire :

[
{x_1,x_2,\ldots,x_T}
]

en un sous-ensemble :

[
{x_{i_1},\ldots,x_{i_K}}.
]

Cette approche correspondrait principalement à du token pruning ou à de la KV-cache eviction.

VELA cherche plutôt à apprendre :

[
{x_1,\ldots,x_T}
\rightarrow
{m_1,\ldots,m_K},
]

où les (m_k) peuvent représenter des abstractions construites à partir de nombreuses observations.

Ainsi :

[
m_k
\neq
\text{embedding d’un token particulier}.
]

Un slot peut résumer ou consolider plusieurs informations historiquement séparées.

Exemple :

[
\text{« une Peugeot rouge appartient à A »}
]

puis plus tard :

[
\text{« la voiture de A est électrique »}
]

peuvent progressivement mettre à jour le même slot :

[
m_{\text{car}}
]

jusqu’à obtenir une représentation consolidée de l’entité correspondante.

VELA passe donc d’une logique de :

[
\boxed{\text{sélection du passé}}
]

à une logique de :

[
\boxed{\text{construction d’un état mémoire}}
]

3. Nouvelle décomposition des opérations mémoire

La mise à jour de la mémoire est désormais explicitement décomposée en plusieurs opérations conceptuelles :

[
\boxed{
\text{ROUTE}
\rightarrow
\text{UPDATE / ALLOCATE}
\rightarrow
\text{CONSOLIDATE}
\rightarrow
\text{FORGET}
}
]

Ces opérations peuvent être implémentées par des modules différentiables.

4. ROUTE — identifier les slots concernés

Lorsqu’une nouvelle représentation (x_t) doit être intégrée à la mémoire, tous les slots ne devraient pas nécessairement être modifiés.

On construit une requête :

[
q_t = q_\theta(x_t)
]

et une clé pour chaque slot :

[
k_t^j = k_\theta(m_{t-1}^j).
]

On calcule ensuite :

[
s_t^j =
\frac{
q_t^\top k_t^j
}{
\sqrt{d}
}.
]

Puis :

[
a_t =
\operatorname{softmax}
\left(
\frac{s_t}{\tau}
\right).
]

Ainsi :

[
a_t^j
]

mesure la pertinence du slot (j) pour la nouvelle information.

Exemple :

[
a_t =
[0.01,0.02,0.03,0.91,0.01,0.02].
]

Le slot 4 est alors le candidat principal à la mise à jour.

La température (\tau) contrôle la concentration du routing.

Une température faible encourage une allocation plus sparse.

5. Sparse routing

Une évolution importante est que VELA ne devrait pas forcément appliquer une mise à jour lourde à tous les slots.

On sélectionne uniquement :

[
S_t =
\operatorname{TopR}(a_t)
]

avec :

[
R \ll K.
]

Par exemple :

[
K=256,
\qquad
R=4.
]

La mémoire peut donc posséder une grande capacité :

[
256\text{ slots}
]

tout en ne mettant à jour que :

[
4\text{ slots}
]

pour une nouvelle information.

Cette architecture s’inspire conceptuellement du sparse routing des Mixture-of-Experts :

[
\boxed{
\text{grande capacité mémoire}
\neq
\text{activation de toute la mémoire}
}
]

Le routing sparse vise deux objectifs :

réduire le coût de calcul des mises à jour ;
encourager la spécialisation des slots.
6. UPDATE — enrichir un concept existant

Si la nouvelle information correspond à un concept déjà présent :

[
a_t^k
]

est élevé pour un slot existant.

On applique alors :

U_\theta(m_{t-1}^k,x_t).
]

Cette mise à jour peut elle-même utiliser :

attention ;
gated update ;
GRU-like update ;
cross-attention ;
MLP conditionnel ;
residual update.

Exemple :

g_t\odot m_{t-1}^k
+
(1-g_t)\odot \tilde m_t^k.
]

Le slot persiste donc dans le temps tout en accumulant progressivement des informations cohérentes.

7. ALLOCATE — créer un nouveau concept

Si aucun slot existant n’est suffisamment pertinent :

[
\max_k a_t^k < \tau_{\mathrm{match}},
]

la nouvelle information peut nécessiter un nouveau slot.

VELA doit donc posséder un mécanisme d’allocation.

On sélectionne par exemple :

un slot inutilisé ;
le slot de plus faible importance ;
le slot le plus ancien ;
le slot le plus facile à fusionner ailleurs.

Puis :

A_\theta(x_t).
]

La mémoire acquiert ainsi progressivement de nouvelles unités conceptuelles.

8. CONSOLIDATE — fusion de slots

Une nouvelle opération centrale est introduite : la consolidation.

Deux slots peuvent initialement avoir été créés indépendamment puis se révéler redondants.

Exemple :

[
m_i
\sim
\text{« voiture rouge de A »},
]

[
m_j
\sim
\text{« voiture électrique appartenant à A »}.
]

Si le système comprend qu’ils décrivent la même entité ou le même facteur latent, il doit pouvoir produire :

C_\theta(m_i,m_j)
]

puis libérer l’un des slots.

On obtient :

[
(m_i,m_j)
\rightarrow
m_{\mathrm{merged}}.
]

La consolidation est fondamentalement différente de l’oubli.

Elle ne supprime pas une information jugée inutile.

Elle réduit la redondance en construisant une représentation plus abstraite.

Cette opération peut être déterminante pour maintenir une mémoire bornée sur des horizons très longs.

9. Détection de la redondance

Une première méthode naïve serait :

\operatorname{cosine}(m_i,m_j).
]

Mais une forte similarité géométrique n’est probablement pas une définition suffisante de la redondance.

Deux slots peuvent être différents en représentation tout en contenant de l’information fonctionnellement équivalente.

Une définition plus pertinente peut être basée sur la redondance prédictive.

Soit :

[
M
]

la mémoire avant fusion et :

[
M^{(i,j)\rightarrow k}
]

la mémoire après fusion de (m_i) et (m_j).

On mesure :

D
\left(
p_\theta(Y\mid M),
p_\theta(Y\mid M^{(i,j)\rightarrow k})
\right).
]

Si :

[
\mathcal L_{\mathrm{merge}}\approx0,
]

alors les deux slots pouvaient probablement être consolidés sans perte significative d’information prédictive.

Cela permet une définition fonctionnelle de la redondance.

10. FORGET — supprimer une information devenue inutile

L’oubli devient une opération explicite :

[
m_k\rightarrow\varnothing.
]

Mais VELA ne doit pas simplement supprimer le slot ayant le plus faible niveau d’activité récente.

L’utilité d’un slot peut dépendre de plusieurs facteurs :

[
u_k =
f(
\text{frequency},
\text{recency},
\text{attention},
\text{surprise},
\text{predictive value},
\text{redundancy},
\text{age}
).
]

L’oubli doit idéalement refléter une estimation de la valeur future probable de l’information.

Une information rarement utilisée peut néanmoins rester cruciale.

Par exemple :

[
\text{code secret donné au début d’une séquence}
]

peut être inutilisé pendant longtemps mais redevenir nécessaire plus tard.

Cela implique que l’oubli ne doit pas être basé uniquement sur la récence.

11. Pression architecturale pour faire émerger des concepts

Un problème central est qu’un modèle sans contrainte particulière peut encoder chaque information de manière distribuée dans tous les slots.

On pourrait obtenir :

0.2m_1
+
0.3m_4
+
0.1m_7
+\cdots
]

au lieu d’une représentation localisée.

VELA doit donc introduire des inductive biases favorisant une organisation plus structurée.

La loss globale peut prendre la forme :

\mathcal L_{\mathrm{task}}
+
\lambda_s\mathcal L_{\mathrm{sparse}}
+
\lambda_p\mathcal L_{\mathrm{persistence}}
+
\lambda_d\mathcal L_{\mathrm{diversity}}
+
\lambda_m\mathcal L_{\mathrm{merge}}
+
\lambda_b\mathcal L_{\mathrm{budget}}
}
]

12. Predictive pressure

La contrainte fondamentale reste :

[
\mathcal L_{\mathrm{task}}.
]

La mémoire doit conserver les informations nécessaires pour estimer ou prédire ce qui est pertinent.

Dans un environnement séquentiel :

[
P(
o_{t+1:\infty}
\mid
o_{1}
)
\approx
P(
o_{t+1:\infty}
\mid
M_t
).
]

Dans un world model :

[
P(
o_{t+1:\infty}
\mid
o_{1},a_{1}
)
\approx
P(
o_{t+1:\infty}
\mid
M_t,a_{t:}
).
]

Cette pression donne du sens aux représentations.

Un slot devient utile s’il capture une information persistante nécessaire pour prédire le futur.

13. Sparse allocation pressure

On peut encourager une faible entropie du routing :

H(a_t)



\sum_k
a_t^k
\log a_t^k.
]

Minimiser cette quantité pousse le modèle vers :

[
x_t
\rightarrow
\text{quelques slots}
]

plutôt que :

[
x_t
\rightarrow
\text{toute la mémoire}.
]

Cela encourage la localité sémantique.

14. Persistence pressure

Les slots ne doivent pas être entièrement réécrits à chaque observation.

On peut introduire :

\sum_k
w_t^k
|m_t^k-m_{t-1}^k|.
]

Une mise à jour doit donc être justifiée par une information nouvelle suffisamment pertinente.

Cela favorise des unités mémoire stables.

15. Diversity / specialization pressure

On veut éviter :

[
m_1\simeq m_2\simeq\cdots\simeq m_K.
]

Une régularisation possible :

\sum_{i\neq j}
\left(
\frac{
m_i^\top m_j
}{
|m_i||m_j|
}
\right)^2.
]

Il faudra cependant éviter de forcer artificiellement tous les concepts à être orthogonaux.

L’objectif est la spécialisation, pas une séparation géométrique arbitraire.

16. Définition fonctionnelle d’un concept

VELA ne doit pas supposer qu’un concept correspond nécessairement à une catégorie humaine explicite.

Une définition plus générale est :

\text{unité mémoire persistante contenant une information prédictivement utile}
}
]

Ainsi un slot peut représenter :

une entité ;
une propriété ;
une relation ;
un objectif ;
un régime de marché ;
une contrainte ;
un état latent dynamique ;
une intention ;
un événement durable.

Cette définition permet d’utiliser la même architecture en langage, finance, robotique ou world modeling.

17. VELA comme hidden state de world model

Dans un world model traditionnel :

[
h_t=f(h_{t-1},o_t,a_{t-1}).
]

VELA propose :

U_\theta(
M_{t-1},
E(o_t),
a_{t-1}
).
]

Le hidden state du world model devient donc :

[
\boxed{
M_t\in\mathbb R^{K\times d_m}
}
]

plutôt qu’un seul vecteur dense.

La transition latente peut ensuite être :

[
p_\phi(
M_{t+1}
\mid
M_t,a_t
).
]

Une variante peut conserver une composante stochastique :

[
S_t=(M_t,z_t)
]

avec :

(M_t) : état persistant structuré ;
(z_t) : innovation/incertitude locale.

Cette formulation rapproche VELA des belief-state world models tout en introduisant une représentation structurée du monde.

18. VELA et object-centric world models

VELA est compatible avec l’idée de représentations object-centric, mais ne se limite pas aux objets.

Les architectures de type Slot Attention ou SlotFormer cherchent souvent :

[
\text{scene}
\rightarrow
{\text{object slots}}.
]

VELA vise plus généralement :

[
\text{history}
\rightarrow
{\text{semantic state slots}}.
]

Un slot pourrait donc représenter un objet, mais aussi :

une relation ;
une intention ;
un régime latent ;
une croyance ;
une contrainte ;
un objectif.

Le terme plus approprié est donc potentiellement :

[
\boxed{\text{concept-centric memory}}
]

plutôt que strictement object-centric memory.

19. Nouveau positionnement pour les LLM

Une clarification majeure concerne l’utilisation de VELA avec les LLM.

VELA ne remplace pas le contexte Transformer pendant le fonctionnement normal du modèle.

Tant que :

[
T<C,
]

où (C) est la taille maximale du contexte, le Transformer fonctionne normalement avec son KV cache complet.

VELA n’intervient que lorsqu’une partie du contexte doit être évincée.

Ainsi :

[
\boxed{
\text{VELA est une mémoire long terme d’overflow}
}
]

et non une alternative permanente au KV cache.

20. Architecture LLM proposée

Le contexte actif devient :

[
\boxed{
[
M_t,
X_{\mathrm{recent}}
]
}
]

où :

[
M_t
]

représente le passé ancien consolidé,

et :

(x_{t-R+1},\ldots,x_t)
]

reste disponible sous forme exacte.

On a :

[
K+R\le C.
]

Donc :

\text{compressed long-term memory}
+
\text{exact recent context}
}
]

Cette distinction est essentielle.

Les informations récentes ou nécessitant une précision exacte restent dans le contexte standard.

VELA absorbe principalement le passé qui serait autrement supprimé.

21. Déclenchement lors de la saturation du contexte

Supposons :

[
C=128,000.
]

VELA peut rester inactif tant que le contexte n’est pas plein.

Lorsque les premiers :

[
B=16,000
]

tokens doivent sortir, on appelle :

U_\theta(
M_0,
X_{1}
).
]

Le contexte devient ensuite :

[
[
M_1,
X_{B+1}
].
]

Lorsque le contexte se remplit de nouveau :

U_\theta(
M_1,
X_{\mathrm{evicted},2}
).
]

Le processus peut continuer indéfiniment :

[
M_0
\rightarrow
M_1
\rightarrow
M_2
\rightarrow
\cdots
]

avec :

[
|M_t|=K.
]

22. Différence avec un résumé textuel

Une stratégie classique consiste à faire :

[
X_{\mathrm{old}}
\rightarrow
\text{LLM}
\rightarrow
\text{summary text}.
]

Cette approche :

utilise un modèle coûteux ;
nécessite une génération autoregressive ;
produit une compression dans l’espace lexical ;
peut accumuler des erreurs de résumé ;
impose de re-tokeniser et re-interpréter le résumé.

VELA propose :

[
X_{\mathrm{old}}
+
M_t
\rightarrow
M_{t+1}
]

directement dans un espace latent.

La compression peut donc être :

plus rapide ;
plus compacte ;
incrémentale ;
non limitée par le langage naturel ;
optimisée spécifiquement pour préserver les comportements du LLM downstream.
23. LLM principal potentiellement inchangé

Une direction importante est de conserver :

[
\boxed{\text{Frozen LLM}}
]

et d’entraîner uniquement VELA.

Le LLM reste donc compatible avec son infrastructure existante.

VELA peut produire des soft tokens :

[
M_t\in\mathbb R^{K\times d_{\mathrm{model}}}
]

qui sont injectés avant le contexte récent :

[
[M_t,X_{\mathrm{recent}}].
]

Une projection peut être utilisée :

[
P_\theta:
\mathbb R^{d_m}
\rightarrow
\mathbb R^{d_{\mathrm{model}}}.
]

Le Transformer continue ensuite son calcul sans modification structurelle majeure.

24. Distillation depuis le contexte complet

Le LLM lui-même fournit un signal d’entraînement naturel.

On prend une séquence qui tient encore dans sa fenêtre :

[
X=
[X_{\mathrm{old}},X_{\mathrm{recent}}].
]

Le teacher voit le contexte complet :

p_{\mathrm{LLM}}
(
y
\mid
X_{\mathrm{old}},X_{\mathrm{recent}}
).
]

VELA compresse :

\operatorname{VELA}(X_{\mathrm{old}}).
]

Le même LLM voit :

[
[M,X_{\mathrm{recent}}],
]

et produit :

[
p_{\mathrm{compressed}}.
]

On peut alors minimiser :

D_{\mathrm{KL}}
\left(
p_{\mathrm{full}}
|
p_{\mathrm{compressed}}
\right)
}
]

VELA apprend ainsi directement :

Quelles informations dois-je conserver pour que le comportement futur du LLM change le moins possible ?

25. Deux niveaux possibles d’intégration LLM
VELA-S — Semantic soft-token memory

VELA produit directement :

[m_1,\ldots,m_K]
]

dans l’espace de représentation du modèle.

Les slots sont ajoutés au contexte.

Avantages :

simplicité ;
compatibilité avec un LLM gelé ;
interprétabilité ;
bon prototype scientifique.

Limite potentielle :

le LLM n’a pas nécessairement été entraîné à interpréter de tels vecteurs arbitraires.

VELA-KV — Semantic state projected into layer-wise KV memory

Une version plus avancée pourrait projeter chaque slot vers les différentes couches :

[
m_k
\rightarrow
{
K_k^{(\ell)},V_k^{(\ell)}
}_{\ell=1}^{L}.
]

On obtient alors une mémoire directement consommable par les mécanismes d’attention internes :

[
M_t
\rightarrow
\text{layer-specific KV memory}.
]

Cette approche pourrait être plus efficace opérationnellement, mais nécessite davantage de modifications du pipeline LLM.

26. Mémoire sémantique et mémoire épisodique

Une mémoire purement sémantique n’est probablement pas suffisante pour tout type d’information.

Certaines informations sont difficilement compressibles :

[
\text{identifiant exact},
]

[
\text{code aléatoire},
]

[
\text{citation exacte},
]

[
\text{nombre précis},
]

[
\text{fragment de code}.
]

VELA pourrait donc devenir hybride :

[
\boxed{
M_t=
M_t^{\mathrm{semantic}}
+
M_t^{\mathrm{episodic}}
}
]

où :

[
M_t^{\mathrm{semantic}}
]

contient des concepts consolidés,

et :

[
M_t^{\mathrm{episodic}}
]

conserve quelques éléments exacts ou récupérables.

Le routeur pourrait également décider :

[
\text{semantic storage}
\quad\text{vs}\quad
\text{exact episodic storage}.
]

27. Nouveau lien avec la compression de contexte

Le problème peut être formulé comme un problème de compression avec contrainte informationnelle.

On cherche :

C_\theta(X_{1})
]

avec :

[
|M_t|\le K,
]

tel que :

[
D
\left(
P(Y_{\mathrm{future}}\mid X_{1}),
P(Y_{\mathrm{future}}\mid M_t)
\right)
]

reste faible.

On peut voir cela comme une forme de compromis rate–distortion :

rate : budget mémoire ;
distortion : perte d’information pertinente pour les prédictions futures.

VELA cherche à apprendre automatiquement la meilleure stratégie de compression pour ce compromis.

28. Différence avec Memformer

Memformer représente un précédent important mais ne correspond pas exactement à la nouvelle définition de VELA.

Memformer utilise une mémoire externe de taille fixe et des slots persistants.

VELA ajoute explicitement :

routing sparse ;
allocation ;
spécialisation ;
merge/consolidation ;
pression architecturale sur la structure des slots ;
utilisation potentielle seulement lors du context overflow ;
distinction contexte récent exact / mémoire long terme ;
objectif de construire des concepts plutôt que de maintenir uniquement des registres latents.

VELA ne doit donc pas être présenté comme :

Memformer avec quelques améliorations.

Une formulation plus précise est :

VELA étudie une mémoire latente structurée et auto-organisée, dont les opérations d’allocation, mise à jour, consolidation et oubli sont apprises sous contrainte prédictive.

29. Différence avec KV pruning

Les méthodes de KV pruning font principalement :

[
\text{keep}
\quad\text{ou}\quad
\text{delete}.
]

VELA ajoute :

[
\text{merge}
]

et :

[
\text{update}.
]

Ainsi :

[
\text{KV pruning}
:
\quad
T\text{ éléments}
\rightarrow
K\text{ éléments sélectionnés},
]

tandis que :

[
\text{VELA}
:
\quad
T\text{ observations}
\rightarrow
K\text{ états latents construits}.
]

30. Différence avec ICAE / AutoCompressors

ICAE et AutoCompressors montrent qu’un contexte peut être compressé dans quelques vecteurs latents.

VELA se différencie par la nature incrémentale et structurée de la mémoire :

U_\theta(
M_n,
X_{\mathrm{evicted}}
).
]

VELA ne cherche pas uniquement à encoder un gros bloc de texte.

Il cherche à maintenir une mémoire persistante sur une série potentiellement infinie de compressions successives.

Le problème scientifique principal devient donc :

[
\boxed{
\text{consolidation répétée sans dérive}
}
]

31. Différence avec LoCoCo / DMC

LoCoCo et DMC cherchent principalement à compresser ou fusionner le KV cache.

VELA vise une couche plus abstraite :

[
\text{KV state}
\rightarrow
\text{semantic state}.
]

L’objectif n’est pas uniquement de réduire le nombre d’entrées KV.

Il est de construire une mémoire représentant les facteurs persistants nécessaires aux prédictions futures.

32. Nouveau critère de réussite

Le succès de VELA ne doit pas être mesuré uniquement par la rétention d’information.

Il faut mesurer simultanément :

[
\boxed{\text{qualité}}
]

[
\boxed{\text{mémoire}}
]

[
\boxed{\text{compute}}
]

[
\boxed{\text{stabilité}}
]

[
\boxed{\text{interprétabilité / spécialisation}}.
]

L’objectif idéal est :

[
\boxed{
\text{meilleure rétention}
+
\text{coût inférieur}
+
\text{mémoire plus structurée}
}
]

33. Nouveau protocole expérimental LLM

Un benchmark futur peut utiliser :

[
\text{Frozen LLM}
+
\text{context overflow}.
]

Baselines :

rolling window ;
truncation ;
résumé textuel par LLM ;
KV pruning ;
ICAE ;
AutoCompressor ;
Activation Beacon ;
LoCoCo ;
DMC ;
VELA.

Faire varier :

128k,
256k,
1M,
10M.
]

Mesurer :

downstream accuracy ;
exact retrieval ;
long-horizon instruction retention ;
factual retention ;
contradiction handling ;
update of previously stored information ;
memory footprint ;
KV memory ;
latency ;
compression FLOPs ;
throughput ;
cumulative degradation après plusieurs consolidations.
34. Test critique : repeated consolidation

Un test particulièrement important est :

[
M_0
\rightarrow
M_1
\rightarrow
M_2
\rightarrow
\cdots
\rightarrow
M_N.
]

Beaucoup de méthodes de compression peuvent fonctionner correctement sur une compression unique.

VELA doit démontrer qu’après :

[
N\gg1
]

compressions successives :

l’information importante persiste ;
les concepts peuvent être mis à jour ;
les contradictions sont résolues ;
les slots ne dégénèrent pas ;
les merges ne détruisent pas progressivement la mémoire.

La résistance à cette dérive devient un critère central.

35. Nouvelle roadmap proposée
Stage 1 — Single-vector belief baseline

Déjà implémenté.

[
o_t
\rightarrow
\text{Embedding}
\rightarrow
\text{GRU}
\rightarrow
z_t.
]

Objectif :

valider le benchmark et l’apprentissage d’un belief state compact.

Stage 2 — Structured slot memory

Introduire :

[
M_t\in\mathbb R^{K\times d_m}.
]

Comparer :

GRU ;
LSTM ;
dense slot memory ;
attention-based slot memory.

Tester :

différentes valeurs de (K) ;
différentes dimensions (d_m) ;
différentes fonctions de readout.
Stage 3 — Sparse routing and specialization

Introduire :

[
p(k\mid x_t).
]

Tester :

dense routing ;
softmax ;
low-temperature softmax ;
Top-(R) routing ;
éventuellement Gumbel-Softmax.

Mesurer :

nombre moyen de slots actifs ;
spécialisation ;
coût ;
qualité prédictive.
Stage 4 — Allocation and learned forgetting

Introduire :

free slots ;
usage score ;
allocation policy ;
deletion ;
replacement.

Créer des environnements où :

certaines informations deviennent obsolètes ;
certaines informations restent longtemps utiles ;
certaines informations reviennent après de longs délais.
Stage 5 — Consolidation / merge

Introduire :

[
C_\theta(m_i,m_j).
]

Tester :

similarity-based merge ;
attention-based merge ;
learned merge score ;
predictive redundancy merge.

Mesurer :

qualité après fusion ;
capacité récupérée ;
erreur induite ;
stabilité sur de longues séquences.
Stage 6 — Representation pressure

Ajouter :

sparse loss ;
persistence loss ;
diversity regularization ;
budget penalty ;
merge consistency ;
éventuellement contrastive/object-centric objectives.

Étudier si les slots développent des représentations stables et spécialisées.

Stage 7 — World-model state

Utiliser :

[
M_t
]

comme hidden state.

Ajouter :

[
p_\phi(M_{t+1}\mid M_t,a_t).
]

Tester :

predictive rollouts ;
belief estimation ;
object permanence ;
hidden-variable tracking ;
planning.
Stage 8 — LLM overflow memory

Prendre un LLM pré-entraîné et gelé.

Conserver :

[
X_{\mathrm{recent}}
]

exactement.

Compresser uniquement :

[
X_{\mathrm{evicted}}.
]

Tester :

U_\theta(M_n,X_{\mathrm{evicted}}).
]

Comparer à rolling context, summaries, KV compression et latent compressors.

Stage 9 — Layer-aware / VELA-KV

Projeter la mémoire sémantique vers les couches Transformer :

[
M_t
\rightarrow
{K^{(\ell)},V^{(\ell)}}_{\ell=1}^{L}.
]

Mesurer si cela améliore :

retention ;
latency ;
downstream accuracy ;
compatibility avec un LLM gelé.
36. Nouvelle formulation synthétique de VELA

Une formulation possible du cœur du projet est désormais :

VELA learns a bounded persistent latent memory that organizes sequential information into a small set of evolving semantic states. New information is sparsely routed to relevant memory slots, which can be updated, allocated, consolidated, or forgotten according to their future predictive value.

Pour les world models :

VELA aims to provide a structured belief state that captures persistent factors of the environment under partial observability.

Pour les LLM :

VELA acts as a learned long-term overflow memory: recent context remains exact, while information that would otherwise be evicted is incrementally consolidated into a fixed-size latent memory consumed alongside the active context.

37. Intuition centrale mise à jour

La différence fondamentale entre les grandes familles peut être résumée ainsi.

Transformer standard

[
\boxed{
\text{store almost everything}
}
]

[
KV_1,\ldots,KV_T.
]

Mémoire :

[
O(T).
]

RNN / GRU

[
\boxed{
\text{compress everything into one state}
}
]

[
h_t.
]

Mémoire :

[
O(1).
]

Mais fort bottleneck informationnel.

KV pruning

[
\boxed{
\text{keep only some past entries}
}
]

Mémoire bornée mais informations supprimées.

Latent compressors

[
\boxed{
\text{compress a block into latent vectors}
}
]

Compression compacte mais généralement moins structurée et moins explicitement dynamique.

VELA

[
\boxed{
\text{construct and maintain a bounded latent model of what matters}
}
]

avec :

[
\boxed{
\text{route}
+
\text{update}
+
\text{allocate}
+
\text{merge}
+
\text{forget}
}
]

et une pression prédictive pour que les slots représentent progressivement des facteurs persistants utiles.

38. Hypothèse scientifique centrale

La nouvelle hypothèse de VELA peut être formulée ainsi :

A bounded memory can retain useful information over much longer horizons if it learns not only what to keep or discard, but how to organize, update and consolidate information into persistent latent concepts.

Cela donne trois hypothèses testables.

H1 — Structured memory

[
M_t
]

avec plusieurs slots conserve mieux les facteurs persistants qu’un seul état récurrent à budget comparable.

H2 — Sparse semantic routing

Mettre à jour seulement les slots pertinents améliore :

l’efficacité ;
la stabilité ;
la spécialisation.
H3 — Learned consolidation

Autoriser la fusion de slots permet de préserver davantage d’information sur de très longs horizons qu’un mécanisme limité à :

[
\text{keep/delete}.
]

39. Ce qui reste explicitement non démontré

Cette note décrit une direction de recherche.

Elle ne suppose pas encore que :

les slots deviennent spontanément interprétables ;
un slot correspond nécessairement à un concept humain ;
sparse routing améliore systématiquement les performances ;
les merges peuvent être appris de manière stable ;
VELA bat les méthodes existantes de compression ;
un LLM gelé peut exploiter efficacement les slots sans adaptation ;
la mémoire reste stable après un nombre arbitraire de consolidations ;
les gains théoriques se traduisent en gains réels de latency ou throughput.

Ces éléments doivent être démontrés expérimentalement.

40. Nouvelle identité du projet

La définition la plus compacte de VELA devient :

[
\boxed{
\textbf{VELA = Value-Aware Evolving Latent Memory}
}
]

ou, si le nom historique Value-Aware Evolving Latent Agent est conservé :

[
\boxed{
\textbf{VELA memory = the persistent structured belief component of the agent}
}
]

Le projet n’étudie donc plus uniquement :

« comment résumer le passé ? »

mais :

« comment un système peut-il construire, maintenir et réorganiser continuellement une mémoire latente compacte de ce qui compte pour comprendre et prédire son monde ? »
