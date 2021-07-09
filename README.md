Provectories using "User Intent" data
==============================

We provide a Python module or importing and projecting user interaction states based on interaction provenance data from Gadhave et al. [1]. The data were captured using the Trrack library [2].

The accompanying website can be accessed here: [Provectories Website](https://provectories.caleydoapp.org/).

![image](https://user-images.githubusercontent.com/22545084/124959232-4c133c80-e01b-11eb-924f-257dd524424f.png)


The supported layouts are
* Topology-driven: **Force-directed layout** using ForceAtlas2
* Attribute-driven: ***t*-SNE** using openTSNE
* Attribute-driven: **UMAP** 
* Attribute-driven: **MDS** using scikit-learn

> To directly fetch interaction provenance data from the user study from the Firebase, an **access token** is needed. This access token is not included in the files. We therefore refer to the owner of the interaction provenance data Kiran Gadhave and Alexander Lex.

[1] K. Gadhave, J. Görtler, O. Deussen, M. Meyer, J. Phillips, and A. Lex. Capturing User Intent when Brushing In Scatterplots. preprint, Open Science Framework, Jan. 2020. doi:  10.31219/osf

[2] Z. T. Cutler, K. Gadhave, and A. Lex.  Trrack: A Library for Provenance Tracking in Web-Based Visualizations. In IEEE Visualizatio nConference (VIS), pp. 116–120, 2020. doi:  10.1109/VIS47514.2020
