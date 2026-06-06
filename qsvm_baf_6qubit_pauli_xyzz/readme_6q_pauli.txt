QSVM 6 qubit — PauliFeatureMap ['X','Y','ZZ']
Input finale: PC1, PC2, PC3, PC4, PC5, email_is_free
Top feature MI: email_is_free (score=0.116937)
Varianza cumulativa PCA(5): 31.45%
Encoding: PauliFeatureMap(paulis=['X','Y','ZZ'], reps=1, entanglement='full')
Differenza rispetto a ZZFeatureMap:
  - ZZFeatureMap usa solo operatori Z e ZZ (encoding esclusivo di fase)
  - PauliFeatureMap con ['X','Y','ZZ'] aggiunge rotazioni Rx e Ry (encoding in ampiezza)
  - Il kernel quantistico risultante copre una varieta piu ricca dello spazio di Hilbert
Nota: il passaggio da ZZFeatureMap a PauliFeatureMap non garantisce automaticamente
piu precisione: il vantaggio va verificato sperimentalmente.
Per usare GPU NVIDIA con Qiskit Aer, installa qiskit-aer-gpu e abilita device='GPU'.
