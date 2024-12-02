# Sistema di Riconoscimento Gestuale basato su Sensori IoT e Reti Neurali

## Introduzione
Il progetto mira a sviluppare un sistema intelligente per il riconoscimento dei gesti utilizzando sensori IoT, con l'obiettivo di interpretare movimenti semplici come UP, DOWN, LEFT e RIGHT. Questo approccio combina l'acquisizione dati tramite sensori di accelerazione e giroscopio con l’addestramento di reti neurali, ottimizzate per catturare informazioni temporali e sequenziali.

---

## Metodologia

### 1. Raccolta e Preprocessing dei Dati
**Frequenza di campionamento:** 0,05 secondi.  

**Dataset:**
- **Dati statici:** 20 file CSV contenenti registrazioni di mano ferma (3 minuti ciascuno).
- **Dati dinamici:** 10 file CSV per ciascun gesto (UP, DOWN, LEFT, RIGHT), ciascuno con durata di 1 minuto.

**Distribuzione complessiva:**
- 90% dei dati rappresentano uno stato fermo.
- 10% dei dati rappresentano movimenti specifici.

**Preprocessing:**
- **Filtraggio:** Eliminazione dei dati di accelerazione per ridurre il rumore, focalizzandosi sui dati del giroscopio.
- **Etichettatura:** Assegnazione manuale delle etichette per ciascun gesto.

---

### 2. Architettura delle Reti Neurali

#### Rete Neurale Convoluzionale (CNN)
- **Caratteristiche:** Applicano filtri spaziali, eccellenti per l’analisi di immagini o pattern statici.
- **Motivo di esclusione:**
  - Inefficace nel trattare dipendenze temporali nei dati sequenziali.
  - Non in grado di catturare le relazioni temporali cruciali nei movimenti.

#### Recurrent Neural Network (RNN)
- **Caratteristiche:** Adatte per dati sequenziali, in grado di memorizzare informazioni temporali.
- **Motivo di esclusione:**
  - Problemi di vanishing gradient rendono difficile l’apprendimento di dipendenze a lungo termine.
  - Limitata nella gestione di gesti complessi e prolungati nel tempo.

#### Long Short-Term Memory (LSTM)
- **Caratteristiche:** Progettate per catturare sia dipendenze a breve che a lungo termine grazie alle celle di memoria dedicate.
- **Vantaggi:**
  - Capacità di memorizzare informazioni temporali cruciali.
  - Migliore gestione delle sequenze lunghe rispetto alle RNN tradizionali.
- **Svantaggi:**
  - Elevata complessità computazionale, utile per dipendenze a lungo termine.

#### Gated Recurrent Unit (GRU)
- **Caratteristiche:** Versione semplificata delle LSTM, con meccanismi di gating che controllano il flusso delle informazioni.
- **Vantaggi:**
  - Prestazioni simili alle LSTM, ma con minore complessità.
  - Adatta per dataset di dimensioni ridotte, riducendo il rischio di overfitting.

**Risultati dell’addestramento:**  
- Accuratezza del 97%.  
- Tendenza all’overfitting, con difficoltà nel riconoscere il primo gesto UP.

#### GRU con Meccanismo di Attenzione
- **Caratteristiche:** Aggiunge un meccanismo di attenzione che attribuisce maggiore peso ai dati rilevanti, migliorando la capacità di catturare gesti distinti.
- **Vantaggi:**
  - Migliora l’accuratezza nel riconoscimento di eventi rari o meno frequenti.
  - Focalizzazione selettiva sui punti chiave della sequenza.

---

## Risultati e Analisi
- **Modello GRU:** Accuratezza del 97%, ma con tendenza all’overfitting sui dati di training, richiedendo ulteriori ottimizzazioni.
- **Modello LSTM:** Comportamento simile alla GRU, con maggiore stabilità nelle dipendenze temporali più lunghe, ma con un costo computazionale più elevato.
- **Modello GRU con attenzione:** Mostra potenziale per migliorare la capacità del modello di rilevare gesti con maggiore precisione, specialmente in contesti con segnali rumorosi o variabili.

---

## Conclusioni e Sviluppi Futuri
Il progetto ha dimostrato che le reti GRU, soprattutto se combinate con meccanismi di attenzione, rappresentano una soluzione efficace per il riconoscimento di gesti basati su dati IoT. Tuttavia, è necessario ottimizzare ulteriormente l’addestramento per ridurre l’overfitting e migliorare la generalizzazione del modello.

**Sviluppi futuri:**
- **Ottimizzazione dei modelli:** Implementare tecniche di regularization e data augmentation.
- **Validazione su dati reali:** Testare il sistema in scenari applicativi concreti, come dispositivi wearable.
- **Integrazione con hardware:** Sviluppare un prototipo funzionale per valutare le prestazioni del sistema in tempo reale.
