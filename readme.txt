Projekt: Rozpoznavani rucne psanych cislic
Predmet: POV
Autori: Ales Raszka, Marek Fiala, Matus Dobrotka
Year: 2016

Projekt je napsan v jazyce Python s vyuzitim knihoven OpenCV 3.1. a Scikit-image 0.18.1.

Pro ovladani je mozno vyuzit tyto parametry:
    
    --train-sklearn-classifier KLASIFIKATOR		
    	Provede natrenovani klasifikatoru z knihovny sklearn na DB MNIST, 
    	jeho ulozeni do souboru a docasne take otestovani na DB MNIST 
    
    --train-opencv-classifier KLASIFIKATOR	
    	Provede natrenovani klasifikatoru z knihovny opencv na DB MNIST, 
    	jeho ulozeni do souboru a docasne take otestovani na DB MNIST 
    
    --classify-mnist-opencv 
    	Klasifikace DB MNIST pomoci klasifikatoru ulozeneho v souboru 'classifier-opencv.pkl'.
    	NEFUNKCNI - OpenCV dosud neumi nacitat klasifikatory ze souboru - jedna se o hlasenou chybu.
    
    --classify-mnist-sklearn
    	Klasifikace DB MNIST pomoci klasifikatoru ulozeneho v souboru 'classifier-sklearn.pkl'.

    --classify-own-opencv OBRAZEK
    	Detekce a klasifikace cisel v obrazku OBRAZEK pomoci klasifikatoru 
    	ulozeneho v souboru 'classifier-opencv.pkl'.
    	NEFUNKCNI - OpenCV dosud neumi nacitat klasifikatory ze souboru - jedna se o hlasenou chybu.

    --classify-own-sklearn OBRAZEK 
    	Detekce a klasifikace cisel v obrazku OBRAZEK pomoci klasifikatoru 
    	ulozeneho v souboru 'classifier-sklearn.pkl'.
                                 

Klasifikatory mozne pouzit misto KLASIFIKATOR jsou nasledujici:
	linear - Linearni SVM
	polynomial - Polynomialni SVM
	k-nearest - K-nejblizsich sousedu
	adaboost - konkretne se jedna o SAMME, funkci pouze pri kombinaci s parametrem -train-sklearn-classifier KLASIFIKATOR	 
	random-forest - Nahodny les


Dodatecne moznosti je mozno nastavit pomoci:
    --mnist
    	Moznost nastaveni polohy DB MNIST

    --block-size CISLO CISLO
    	Moznost nastaveni velikosti bloku pro vypocet HOG - napriklad 7 7

    --cell-size CISLO CISLO
    	Moznost nastaveni velikosti bunky pro vypocet HOG - napriklad 1 1

    --bins CISLO
    	Moznost nastaveni poctu binu pro HOG

