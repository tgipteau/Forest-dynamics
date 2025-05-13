# LS2N

Les scripts présents dans ce dépôt sont des outils de simulation d'évolution de forêt.<br>

### Les modèles

On distingue deux modèles :
<ul>
    <li> Un modèle dit "à deux équations", qui modélise l'évolution de la densité de population d'une espèce
    d'arbre et de ses graines. On ajoute à ce processus déterministe (EDP) un dérèglement stochastique par 
    l'apparition de feux de forêt aléatoires.</li>
    <li> Un modèle dit "à quatre équations", qui ajoute au modèle précédent une deuxième espèce d'arbres et ses graines.
    On ajoute alors un facteur de compétition inter-espèces, ainsi qu'un effet spatial (préférence pour le nord ou
    le sud en fonction de l'espèce). </li>
</ul>


### Contenus de ce dépôt - Partie "visualisation"

<b>Toundra2</b> : utilise le modèle à quatre équations (deux espèces d'arbres). Utilise FreeFem++ (logiciel externe).<br>
Ce Fiso est destiné à une utilisation en local après installation de FreeFem++. Comprend une interface utilisateur<br>
pour lancer des simulations et les charger (<i>user-friendly</i>).<br>

<b>Toundra3</b> : utilise le modèle à deux équations (une seule espèce d'arbre). C'est un programme entièrement écrit<br>
en Python, et donc portable sur une plateforme comme PythonAnywhere. Ne comprend pas d'interface utilisateur<br>
pour configurer la simulation : la configuration se fait par l'édition d'un fichier <i>.yaml</i>.<br>

<b>Toundra4</b> : utilise le modèle à quatre équations (deux espèces d'arbres). C'est un programme entièrement écrit<br>
en Python, et donc portable sur une plateforme comme PythonAnywhere.  Ne comprend pas d'interface utilisateur<br>
pour configurer la simulation : la configuration se fait par l'édition d'un fichier <i>.yaml</i>.<br>
<br>

___

### Partie "chaînes de markov"

Je travaile maintenant sur une modélisation en chaîne de markov du procédé hybride edp/stochastique.


