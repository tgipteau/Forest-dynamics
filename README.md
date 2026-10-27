# Dynamique de forêts

Les scripts présents dans ce dépôt sont des outils de simulation d'évolution de forêt réalisés pour le LS2N à Nantes.<br>

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


### Contenus de ce dépôt 

<b> Un article de recherche </b> sur le modèle utilisé, ainsi qu'une étude d'un modèle de markov hybride de simplification.

<b>Toundra2</b> : utilise le modèle à quatre équations (deux espèces d'arbres). Utilise FreeFem++ (logiciel externe).<br>
Destiné à une utilisation en local après installation de FreeFem++. Comprend une interface utilisateur<br>
pour lancer des simulations et les charger.<br>

<b>Toundra3</b> : utilise le modèle à deux équations (une seule espèce d'arbre). Entièrement Python.<br>
Ne comprend pas d'interface utilisateur pour configurer la simulation : <br>
la configuration se fait par l'édition d'un fichier <i>.yaml</i>.<br>

<b>Toundra4</b> : utilise le modèle à quatre équations (deux espèces d'arbres). Entièrement Python. <br>
Ne comprend pas d'interface utilisateur pour configurer la simulation : <br>
la configuration se fait par l'édition d'un fichier <i>.yaml</i>.<br>



