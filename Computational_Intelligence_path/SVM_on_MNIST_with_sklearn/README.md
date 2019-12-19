2η Εργασία στα Νευρωνικά Δίκτυα και Βαθιά Μάθηση









Κωνσταντίνος Νικόπουλος 2782







Σε αυτή την εργασία υλοποίησα SVMs με το sklearn στην MNIST για την αναγνώριση μονών και ζυγών αριθμών.







1. Preprocess dataset



Αρχικά χρησιμοποίησα ένα τμήμα του dataset.

new shape of data: (10000, 784) (2000, 784)

new shape of labels: (10000,) (2000,)

Έπειτα επεξεργάστηκα τα δεδομένα και εφάρμοσα PCA για τη μείωση των διαστάσεων απο 784 σε 100.



Επομένως το dataset έρχεται στη μορφή:

new shape of data: (10000, 100) (2000, 100)

new shape of labels: (10000,) (2000,)




















2. Parameters



2.1 Kernels



Οι kernels που χρησιμοποίηκαν είναι ο linear και ο polynomial με βαθμούς 2 και 3.



2.2 C



Η  C είναι μια παράμετρος κανονικοπίησης που ελέγχει το σφάλμα.  Διαμορφώνει το μέγεθος του margin, μικρό C ->  μεγάλο margin, μεγάλο C ->  μικρό margin. 

Οι τιμές που χρησιμοποίηκαν είναι 1, 10 και 100.







3. SVMs 

Με Grid search βρέθηκαν οι καλύτερες παράμετροι από τις παραπάνω (για επαλήθευση).



3.1 Final SVM

Το SVM με τις καλύτερες παραμέτρους από τις παραπάνω είναι με kernel polynomial 3 και C 1. 



3.2 Results

Τα accuracy score όλων των μοντέλων είναι:

kernel: linear, C=1   : 0.874

kernel: linear, C=10   : 0.873

kernel: linear, C=100   : 0.873

kernel: polynomial, degree=2, C=1   : 0.9625

kernel: polynomial, degree=2, C=10   : 0.9595

kernel: polynomial, degree=2, C=100   : 0.959

kernel: polynomial, degree=3, C=1   : 0.972

kernel: polynomial, degree=3, C=10   : 0.971

kernel: polynomial, degree=3, C=100   : 0.971

Οπότε το καλύτερο με παραμέτρους  kernel polynomial 3 και C 1:

Το accuracy score του στα δεδομένα test είναι 0.972. 

Ο χρόνος εκπαίδευσής του είναι 6.202604s.







4. Comparison with Nearest Neighbor and Nearest Class Centroid

Χρησιμοποιήσα τον κώδικα από την ενδιάμεση εργασία και εφάρμοσα τους αλγορίθμους στην MNIST. 



4.1  K=1 Nearest Neighbor

Με accuracy 0.971 είναι πολύ κοντά στο μοντέλο του SVM.



4.2 K=3 Nearest Neighbor

Ίδιο με k=1 με accuracy 0.971. Είναι πολύ κοντά στο μοντέλο του SVM.



4.3  Nearest Class Centroid

Με accuracy 0.7835 προκύπτει πως έχει το μικρότερο accuracy.



4.4 Recap

Επομένως  ο καλύτερος αλγόριθμος για το πρόβλημα της MNIST (μονών και ζυγών) είναι το SVM με kernel polynomial, degree 3 και C 1. 







































