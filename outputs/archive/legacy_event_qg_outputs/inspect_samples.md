# MAVEN-ERE Data Inspection

**Split**: train | **Docs inspected**: 10

**Top-level keys**: ['TIMEX', 'causal_relations', 'events', 'id', 'sentences', 'subevent_relations', 'temporal_relations', 'title', 'tokens']

---

## Sample 1: Expedition of the Thousand
**Doc ID**: 364ed14fc610df6e25a2f446e2b2d2ab

### Sentences (7 total)
  [0] The Expedition of the Thousand (Italian "Spedizione dei Mille") was an event of the Italian Risorgimento that took place in 1860.
  [1] A corps of volunteers led by Giuseppe Garibaldi sailed from Quarto, near Genoa (now Quarto dei Mille) and landed in Marsala, Sicily, in order to conquer the Kingdom of the Two Sicilies, ruled by the H
  [2] The project was an ambitious and risky venture aiming to conquer, with a thousand men, a kingdom with a larger regular army and a more powerful navy.
  [3] The expedition was a success and concluded with a plebiscite that brought Naples and Sicily into the Kingdom of Sardinia, the last territorial conquest before the creation of the Kingdom of Italy on 1
  [4] The sea venture was the only desired action that was jointly decided by the "four fathers of the nation" Giuseppe Mazzini, Giuseppe Garibaldi, Victor Emmanuel II, and Camillo Cavour, pursuing divergen
  [5] However, the Expedition was instigated by Francesco Crispi, who utilized his political influence to bolster the Italian unification project.
  [6] The various groups participated in the expedition for a variety of reasons: for Garibaldi, it was to achieve a united Italy; to the Sicilian bourgeoisie, an independent Sicily as part of the kingdom o

### Events (21 total)
  - EVENT_e2cbbfb4a5ab65856ccc7057443120e0 | type=Control | trigger='ruled' | sent_id=1 | offset=[38, 39]
  - EVENT_0b0c5d1268cc135278489b7f0004c0c3 | type=Achieve | trigger='achieve' | sent_id=6 | offset=[19, 20]
  - EVENT_e71684bf9f6e7b50bf5316982a640c15 | type=Creating | trigger='creation' | sent_id=3 | offset=[27, 28]
  - EVENT_f535eb21f8f958a04cbc69c1d1521186 | type=Self_motion | trigger='Expedition' | sent_id=0 | offset=[1, 2]
  - EVENT_f535eb21f8f958a04cbc69c1d1521186 | type=Self_motion | trigger='venture' | sent_id=2 | offset=[7, 8]
  - EVENT_f535eb21f8f958a04cbc69c1d1521186 | type=Self_motion | trigger='expedition' | sent_id=3 | offset=[1, 2]
  - EVENT_f535eb21f8f958a04cbc69c1d1521186 | type=Self_motion | trigger='Expedition' | sent_id=5 | offset=[3, 4]
  - EVENT_f535eb21f8f958a04cbc69c1d1521186 | type=Self_motion | trigger='expedition' | sent_id=6 | offset=[6, 7]
  - EVENT_d2587256a67ce75ea9d9d864288d822c | type=Motion | trigger='landed' | sent_id=1 | offset=[21, 22]
  - EVENT_e12e909ff79793dc2b7b534d37685e9c | type=Process_start | trigger='instigated' | sent_id=5 | offset=[5, 6]
  - EVENT_f652152ef7245a6f5c27396a2b1964e1 | type=Process_end | trigger='concluded' | sent_id=3 | offset=[6, 7]
  - EVENT_15451e1aabcb291d411af6a3d7e32911 | type=Cause_to_amalgamate | trigger='unification' | sent_id=5 | offset=[19, 20]
  - EVENT_6912f3e89bb068b0fdd78948b8cb9be2 | type=Aiming | trigger='aiming' | sent_id=2 | offset=[8, 9]
  - EVENT_93e0350c5f49e6ddb79b2091fda2012a | type=Bringing | trigger='brought' | sent_id=3 | offset=[11, 12]
  - EVENT_223bc4cd361349a0bd7fbd0065eb0f7d | type=Participation | trigger='participated' | sent_id=6 | offset=[3, 4]
  - EVENT_e92e9b331fe3220ec0a33d8bd2489fd0 | type=Ratification | trigger='plebiscite' | sent_id=3 | offset=[9, 10]
  - EVENT_c027e659d7fe424a0a57ecbe35b3a7f9 | type=Conquering | trigger='conquer' | sent_id=1 | offset=[30, 31]
  - EVENT_c027e659d7fe424a0a57ecbe35b3a7f9 | type=Conquering | trigger='conquer' | sent_id=2 | offset=[10, 11]
  - EVENT_c027e659d7fe424a0a57ecbe35b3a7f9 | type=Conquering | trigger='conquest' | sent_id=3 | offset=[24, 25]
  - EVENT_931d31a625729392d27925e697762d5f | type=Self_motion | trigger='venture' | sent_id=4 | offset=[2, 3]
  - EVENT_931d31a625729392d27925e697762d5f | type=Self_motion | trigger='action' | sent_id=4 | offset=[7, 8]
  - EVENT_4381e133e1cf65fa2e5bc1767ed1d0db | type=Using | trigger='utilized' | sent_id=5 | offset=[11, 12]
  - EVENT_5d5130ba19c9706c4fff53d34225fb1c | type=Process_start | trigger='took place' | sent_id=0 | offset=[21, 23]
  - EVENT_8d2fbddb83f3c50eba8e24e215101547 | type=Deciding | trigger='decided' | sent_id=4 | offset=[11, 12]
  - EVENT_5445faf0caf78e3dcc1dc88e3554fac8 | type=Process_end | trigger='end' | sent_id=6 | offset=[49, 50]
  - EVENT_451b7cde13d2b8c21426db027c51096f | type=Self_motion | trigger='sailed' | sent_id=1 | offset=[8, 9]
  - EVENT_040b46602889e455df7b4327137034e9 | type=Cause_to_make_progress | trigger='bolster' | sent_id=5 | offset=[16, 17]
  - EVENT_562d52136def8fdfaa89dc3852695404 | type=Dispersal | trigger='distribution' | sent_id=6 | offset=[46, 47]

### causal_relations (2 entries)
  ** CAUSE: 0 pairs
  ** PRECONDITION: 14 pairs
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_c027e659d7fe424a0a57ecbe35b3a7f9']
    -> ['EVENT_f535eb21f8f958a04cbc69c1d1521186', 'EVENT_5445faf0caf78e3dcc1dc88e3554fac8']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_0b0c5d1268cc135278489b7f0004c0c3']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_931d31a625729392d27925e697762d5f']
    -> ['EVENT_f535eb21f8f958a04cbc69c1d1521186', 'EVENT_0b0c5d1268cc135278489b7f0004c0c3']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_451b7cde13d2b8c21426db027c51096f']
    -> ['EVENT_8d2fbddb83f3c50eba8e24e215101547', 'EVENT_931d31a625729392d27925e697762d5f']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_562d52136def8fdfaa89dc3852695404']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_f535eb21f8f958a04cbc69c1d1521186']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_d2587256a67ce75ea9d9d864288d822c']
  ... (showing 10 of 14)

### temporal_relations (6 entries)
  ** BEFORE: 121 pairs
    -> ['EVENT_8d2fbddb83f3c50eba8e24e215101547', 'EVENT_f535eb21f8f958a04cbc69c1d1521186']
    -> ['EVENT_e92e9b331fe3220ec0a33d8bd2489fd0', 'EVENT_562d52136def8fdfaa89dc3852695404']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_93e0350c5f49e6ddb79b2091fda2012a']
    -> ['EVENT_d2587256a67ce75ea9d9d864288d822c', 'EVENT_0b0c5d1268cc135278489b7f0004c0c3']
    -> ['EVENT_e12e909ff79793dc2b7b534d37685e9c', 'EVENT_c027e659d7fe424a0a57ecbe35b3a7f9']
    -> ['EVENT_8d2fbddb83f3c50eba8e24e215101547', 'EVENT_f652152ef7245a6f5c27396a2b1964e1']
    -> ['EVENT_d2587256a67ce75ea9d9d864288d822c', 'EVENT_562d52136def8fdfaa89dc3852695404']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_931d31a625729392d27925e697762d5f']
    -> ['TIME_c61b2c2b8b8c6656a1cc8443fed8c58a', 'EVENT_e71684bf9f6e7b50bf5316982a640c15']
    -> ['EVENT_040b46602889e455df7b4327137034e9', 'EVENT_c027e659d7fe424a0a57ecbe35b3a7f9']
  ** OVERLAP: 4 pairs
    -> ['EVENT_e2cbbfb4a5ab65856ccc7057443120e0', 'EVENT_5d5130ba19c9706c4fff53d34225fb1c']
    -> ['EVENT_e2cbbfb4a5ab65856ccc7057443120e0', 'EVENT_931d31a625729392d27925e697762d5f']
    -> ['EVENT_e2cbbfb4a5ab65856ccc7057443120e0', 'TIME_c61b2c2b8b8c6656a1cc8443fed8c58a']
    -> ['EVENT_e2cbbfb4a5ab65856ccc7057443120e0', 'EVENT_f535eb21f8f958a04cbc69c1d1521186']
  ** CONTAINS: 39 pairs
    -> ['TIME_c61b2c2b8b8c6656a1cc8443fed8c58a', 'EVENT_5d5130ba19c9706c4fff53d34225fb1c']
    -> ['TIME_c61b2c2b8b8c6656a1cc8443fed8c58a', 'EVENT_d2587256a67ce75ea9d9d864288d822c']
    -> ['EVENT_e12e909ff79793dc2b7b534d37685e9c', 'EVENT_4381e133e1cf65fa2e5bc1767ed1d0db']
    -> ['TIME_c61b2c2b8b8c6656a1cc8443fed8c58a', 'EVENT_223bc4cd361349a0bd7fbd0065eb0f7d']
    -> ['EVENT_5d5130ba19c9706c4fff53d34225fb1c', 'EVENT_f652152ef7245a6f5c27396a2b1964e1']
    -> ['EVENT_931d31a625729392d27925e697762d5f', 'EVENT_451b7cde13d2b8c21426db027c51096f']
    -> ['EVENT_6912f3e89bb068b0fdd78948b8cb9be2', 'EVENT_f535eb21f8f958a04cbc69c1d1521186']
    -> ['EVENT_f535eb21f8f958a04cbc69c1d1521186', 'EVENT_451b7cde13d2b8c21426db027c51096f']
    -> ['EVENT_f535eb21f8f958a04cbc69c1d1521186', 'EVENT_d2587256a67ce75ea9d9d864288d822c']
    -> ['EVENT_e2cbbfb4a5ab65856ccc7057443120e0', 'EVENT_451b7cde13d2b8c21426db027c51096f']
  ** SIMULTANEOUS: 3 pairs
    -> ['EVENT_f535eb21f8f958a04cbc69c1d1521186', 'EVENT_5d5130ba19c9706c4fff53d34225fb1c']
    -> ['EVENT_f535eb21f8f958a04cbc69c1d1521186', 'EVENT_931d31a625729392d27925e697762d5f']
    -> ['EVENT_931d31a625729392d27925e697762d5f', 'EVENT_5d5130ba19c9706c4fff53d34225fb1c']
  ** ENDS-ON: 1 pairs
    -> ['EVENT_e2cbbfb4a5ab65856ccc7057443120e0', 'EVENT_f652152ef7245a6f5c27396a2b1964e1']
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 168)

### subevent_relations (2 entries)
  -> ['EVENT_f535eb21f8f958a04cbc69c1d1521186', 'EVENT_d2587256a67ce75ea9d9d864288d822c']
  -> ['EVENT_f535eb21f8f958a04cbc69c1d1521186', 'EVENT_451b7cde13d2b8c21426db027c51096f']

---

## Sample 2: Murder of Leigh Leigh
**Doc ID**: 0371bbf116422f8e3a0a853bdd1962aa

### Sentences (14 total)
  [0] The murder of Leigh Leigh, born Leigh Rennea Mears, occurred on 3 November 1989 while she was attending a 16-year-old boy's birthday party at Stockton Beach, New South Wales, on the east coast of Aust
  [1] The 14-year-old girl from Fern Bay was assaulted by a group of boys after she returned distressed from a sexual encounter on the beach that a reviewing judge later called non-consensual.
  [2] After being kicked and spat on by the group, Leigh left the party.
  [3] Her naked body was found in the sand dunes nearby the following morning, with severe genital damage and a crushed skull.
  [4] Matthew Grant Webster, an 18-year-old who acted as a bouncer at the event, pleaded guilty to her murder and was sentenced to 20 years in prison with a 14-year non-parole period.
  [5] He was released on parole in June 2004, after serving 14½ years.
  [6] Guy Charles Wilson, the other bouncer and only other person aged over 18 at the party, pleaded guilty to assault; a third male (aged 15) pleaded guilty to having sex with a minor.
  [7] The investigation of Leigh's murder proved controversial, however, as several people who admitted to various crimes, including assaulting Leigh, were never charged; nor was anyone ever charged with he
  [8] Webster's confession did not match the forensic evidence.
  [9] The murder investigation was reviewed by the New South Wales Crime Commission in 1996, and by the Police Integrity Commission in 1998, with the latter recommending the dismissal of the detective in ch
  [10] Leigh's murder received considerable attention in the media.
  [11] Initially focusing on her sexual assault and murder, media attention later concentrated more on the lack of parental supervision and the drugs and alcohol at the party, and on Leigh's sexuality.
  [12] The media coverage of the murder has been cited as an example of blaming the victim.
  [13] Leigh's murder inspired a theatrical play entitled "A Property of the Clan", which was later revised and renamed "Blackrock", as well as a feature film of the same name.

### Events (31 total)
  - EVENT_36fe15f2ecceb14abdedc904976ed5aa | type=Convincing | trigger='recommending' | sent_id=9 | offset=[27, 28]
  - EVENT_9424f1b2452e2a1b4550890e6d4d4605 | type=Name_conferral | trigger='renamed' | sent_id=13 | offset=[21, 22]
  - EVENT_40c838f87021aea782979dda7d1e218c | type=Judgment_communication | trigger='charged' | sent_id=7 | offset=[26, 27]
  - EVENT_5389810ea556a9d008dd6d5b58e53751 | type=Self_motion | trigger='kicked' | sent_id=2 | offset=[2, 3]
  - EVENT_5ac6fb32c1bcdbe3db31fe40c09bc9bf | type=Request | trigger='pleaded' | sent_id=6 | offset=[18, 19]
  - EVENT_ad5f54e6c5e99b350440a6522904c7d6 | type=Aiming | trigger='focusing' | sent_id=11 | offset=[1, 2]
  - EVENT_8139a667841b690aa2836529b1aebea4 | type=Assistance | trigger='serving' | sent_id=5 | offset=[10, 11]
  - EVENT_77c5bbce3ff0b62b706cf77c998a64cd | type=Attack | trigger='assaulted' | sent_id=1 | offset=[7, 8]
  - EVENT_0e6724982bfe0fc308607e61bb8dfd74 | type=Legal_rulings | trigger='sentenced' | sent_id=4 | offset=[22, 23]
  - EVENT_f53be1f24a44da8501d7c9e0fc32d9d1 | type=Name_conferral | trigger='called' | sent_id=1 | offset=[29, 30]
  - EVENT_047960b77032ca09f62480b46c3a68b4 | type=Adducing | trigger='cited' | sent_id=12 | offset=[8, 9]
  - EVENT_d80b82105bba0dd4811979d0140c810e | type=Coming_to_be | trigger='occurred' | sent_id=0 | offset=[11, 12]
  - EVENT_9b603ed7f3b8af014a6bcb95c12fcd34 | type=Request | trigger='pleaded' | sent_id=4 | offset=[15, 16]
  - EVENT_976afc9eb811d6d993fe4cfe7a856ea8 | type=Departing | trigger='left' | sent_id=2 | offset=[11, 12]
  - EVENT_7d19d91d981674033870cbd6fc303615 | type=Judgment_communication | trigger='charged' | sent_id=7 | offset=[32, 33]
  - EVENT_828dc91e6e786fa11cd961db5561d372 | type=Bodily_harm | trigger='crushed' | sent_id=3 | offset=[20, 21]
  - EVENT_6943c6cff9c90f9c274888d8e406ba6d | type=Competition | trigger='match' | sent_id=8 | offset=[5, 6]
  - EVENT_484a71e3aaf678480ca0f99747a4eba0 | type=Damaging | trigger='damage' | sent_id=3 | offset=[17, 18]
  - EVENT_9183176dad5d9f3b855d12150420efa0 | type=Aiming | trigger='concentrated' | sent_id=11 | offset=[12, 13]
  - EVENT_473078a1254cacf3c9060ccb1a0b1d0e | type=Preventing_or_letting | trigger='entitled' | sent_id=13 | offset=[7, 8]
  - EVENT_afb6ae793bc303e958538be555fad91c | type=Criminal_investigation | trigger='investigation' | sent_id=9 | offset=[2, 3]
  - EVENT_afb6ae793bc303e958538be555fad91c | type=Criminal_investigation | trigger='investigation' | sent_id=9 | offset=[37, 38]
  - EVENT_1050c9dc67ba5fe1b77e8288e7e2089f | type=Request | trigger='pleaded' | sent_id=6 | offset=[30, 31]
  - EVENT_93be1937165e85f026f8367b3a59c236 | type=Influence | trigger='inspired' | sent_id=13 | offset=[3, 4]
  - EVENT_f6785336c5d36e42da51edb8e1ef20e0 | type=Research | trigger='reviewed' | sent_id=9 | offset=[4, 5]
  - EVENT_f1e73d28d30014d0f4d274a30bbf0c59 | type=Change | trigger='revised' | sent_id=13 | offset=[19, 20]
  - EVENT_f69dc296dec6e31675b1a184aca59218 | type=Know | trigger='found' | sent_id=3 | offset=[4, 5]
  - EVENT_1e1f24501a9f61b0e2ace8eb26c58f97 | type=Committing_crime | trigger='murder' | sent_id=0 | offset=[1, 2]
  - EVENT_1e1f24501a9f61b0e2ace8eb26c58f97 | type=Committing_crime | trigger='murder' | sent_id=4 | offset=[19, 20]
  - EVENT_1e1f24501a9f61b0e2ace8eb26c58f97 | type=Committing_crime | trigger='murder' | sent_id=7 | offset=[5, 6]
  - EVENT_1e1f24501a9f61b0e2ace8eb26c58f97 | type=Committing_crime | trigger='murder' | sent_id=11 | offset=[7, 8]
  - EVENT_1e1f24501a9f61b0e2ace8eb26c58f97 | type=Committing_crime | trigger='murder' | sent_id=12 | offset=[5, 6]
  - EVENT_aa7d159fbcf3627c741ab98fcb00c155 | type=Releasing | trigger='released' | sent_id=5 | offset=[2, 3]
  - EVENT_90adcba4b96ea5f6cd9f93d5e5111e32 | type=Judgment_communication | trigger='blaming' | sent_id=12 | offset=[13, 14]
  - EVENT_7750c883b0e836cf2108476520071012 | type=Receiving | trigger='received' | sent_id=10 | offset=[3, 4]

### causal_relations (2 entries)
  ** CAUSE: 0 pairs
  ** PRECONDITION: 13 pairs
    -> ['EVENT_d80b82105bba0dd4811979d0140c810e', 'EVENT_7750c883b0e836cf2108476520071012']
    -> ['EVENT_473078a1254cacf3c9060ccb1a0b1d0e', 'EVENT_f1e73d28d30014d0f4d274a30bbf0c59']
    -> ['EVENT_5389810ea556a9d008dd6d5b58e53751', 'EVENT_976afc9eb811d6d993fe4cfe7a856ea8']
    -> ['EVENT_1e1f24501a9f61b0e2ace8eb26c58f97', 'EVENT_7750c883b0e836cf2108476520071012']
    -> ['EVENT_473078a1254cacf3c9060ccb1a0b1d0e', 'EVENT_9424f1b2452e2a1b4550890e6d4d4605']
    -> ['EVENT_9b603ed7f3b8af014a6bcb95c12fcd34', 'EVENT_0e6724982bfe0fc308607e61bb8dfd74']
    -> ['EVENT_d80b82105bba0dd4811979d0140c810e', 'EVENT_93be1937165e85f026f8367b3a59c236']
    -> ['EVENT_d80b82105bba0dd4811979d0140c810e', 'EVENT_f69dc296dec6e31675b1a184aca59218']
    -> ['EVENT_1e1f24501a9f61b0e2ace8eb26c58f97', 'EVENT_93be1937165e85f026f8367b3a59c236']
    -> ['EVENT_1e1f24501a9f61b0e2ace8eb26c58f97', 'EVENT_f69dc296dec6e31675b1a184aca59218']
  ... (showing 10 of 13)

### temporal_relations (6 entries)
  ** BEFORE: 371 pairs
    -> ['EVENT_f69dc296dec6e31675b1a184aca59218', 'EVENT_f1e73d28d30014d0f4d274a30bbf0c59']
    -> ['EVENT_1e1f24501a9f61b0e2ace8eb26c58f97', 'TIME_7ad0f1d261a3cf985c293ddcf72aef2e']
    -> ['EVENT_473078a1254cacf3c9060ccb1a0b1d0e', 'EVENT_aa7d159fbcf3627c741ab98fcb00c155']
    -> ['EVENT_828dc91e6e786fa11cd961db5561d372', 'EVENT_7d19d91d981674033870cbd6fc303615']
    -> ['EVENT_5389810ea556a9d008dd6d5b58e53751', 'EVENT_9424f1b2452e2a1b4550890e6d4d4605']
    -> ['TIME_626612ce2546f23317fe9c49b937db29', 'TIME_c5944a5050643536c536fa04f9b29442']
    -> ['EVENT_5ac6fb32c1bcdbe3db31fe40c09bc9bf', 'EVENT_1050c9dc67ba5fe1b77e8288e7e2089f']
    -> ['EVENT_7d19d91d981674033870cbd6fc303615', 'EVENT_afb6ae793bc303e958538be555fad91c']
    -> ['EVENT_f6785336c5d36e42da51edb8e1ef20e0', 'TIME_c5944a5050643536c536fa04f9b29442']
    -> ['EVENT_1050c9dc67ba5fe1b77e8288e7e2089f', 'EVENT_36fe15f2ecceb14abdedc904976ed5aa']
  ** OVERLAP: 2 pairs
    -> ['EVENT_8139a667841b690aa2836529b1aebea4', 'TIME_c5944a5050643536c536fa04f9b29442']
    -> ['TIME_7ad0f1d261a3cf985c293ddcf72aef2e', 'TIME_c5944a5050643536c536fa04f9b29442']
  ** CONTAINS: 44 pairs
    -> ['TIME_7ad0f1d261a3cf985c293ddcf72aef2e', 'EVENT_afb6ae793bc303e958538be555fad91c']
    -> ['TIME_626612ce2546f23317fe9c49b937db29', 'EVENT_77c5bbce3ff0b62b706cf77c998a64cd']
    -> ['TIME_d9e52a90f6b0e353dadf234cf40befe1', 'EVENT_7750c883b0e836cf2108476520071012']
    -> ['TIME_d9e52a90f6b0e353dadf234cf40befe1', 'EVENT_93be1937165e85f026f8367b3a59c236']
    -> ['TIME_d9e52a90f6b0e353dadf234cf40befe1', 'EVENT_90adcba4b96ea5f6cd9f93d5e5111e32']
    -> ['TIME_c5944a5050643536c536fa04f9b29442', 'EVENT_aa7d159fbcf3627c741ab98fcb00c155']
    -> ['TIME_626612ce2546f23317fe9c49b937db29', 'EVENT_976afc9eb811d6d993fe4cfe7a856ea8']
    -> ['EVENT_8139a667841b690aa2836529b1aebea4', 'EVENT_7d19d91d981674033870cbd6fc303615']
    -> ['TIME_7ad0f1d261a3cf985c293ddcf72aef2e', 'EVENT_7d19d91d981674033870cbd6fc303615']
    -> ['EVENT_8139a667841b690aa2836529b1aebea4', 'EVENT_5ac6fb32c1bcdbe3db31fe40c09bc9bf']
  ** SIMULTANEOUS: 4 pairs
    -> ['EVENT_8139a667841b690aa2836529b1aebea4', 'TIME_7ad0f1d261a3cf985c293ddcf72aef2e']
    -> ['EVENT_7d19d91d981674033870cbd6fc303615', 'EVENT_40c838f87021aea782979dda7d1e218c']
    -> ['EVENT_d80b82105bba0dd4811979d0140c810e', 'EVENT_1e1f24501a9f61b0e2ace8eb26c58f97']
    -> ['EVENT_f6785336c5d36e42da51edb8e1ef20e0', 'EVENT_afb6ae793bc303e958538be555fad91c']
  ** ENDS-ON: 0 pairs
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 421)

---

## Sample 3: Hurricane Jerry (1989)
**Doc ID**: 097b86ef2a4c5d21037a4c6b47b3a165

### Sentences (20 total)
  [0] Hurricane Jerry caused minor damage in Texas and flash flooding in Kentucky and Virginia in October 1989.
  [1] The fourteenth tropical cyclone, tenth named storm of the season, Jerry developed from a tropical wave in the Bay of Campeche on October 12.
  [2] Initially a tropical depression, the system moved north-northwestward across the Gulf of Mexico and strengthened into Tropical Storm Jerry early on the following day.
  [3] Jerry continuously deepened until October 14 and then maintained intensity while curving northeastward and briefly decelerating.
  [4] Later that day, the storm re-curved north-northwestward.
  [5] Jerry began to intensify on October 15 and soon became a Category 1 hurricane on the Saffir–Simpson hurricane wind scale.
  [6] Early on October 16, Jerry made landfall on Galveston Island, Texas with winds of .
  [7] Less than six hours later, Jerry weakened to a tropical storm and then a tropical depression shortly thereafter.
  [8] Late on October 16, Jerry was absorbed by a frontal system while situated over southwestern Arkansas.
  [9] Storm surge and rough surf along the coast of Texas destroyed a section of Texas State Highway 87, which was never repaired.
  [10] Due to strong winds, about 52,000 homes and businesses were left without electricity, most of them in the Galveston area.
  [11] Many homes, businesses, and buildings were inflicted damage because of strong winds and three tornadoes spawned by the storm.
  [12] Despite the issuance of a hurricane warning just eight hours prior to landfall, Jerry caused only three fatalities in Texas, possibly due to the storm's small size; a car fell over the Galveston Seawa
  [13] Minor wind and coastal flood damage was reported in Louisiana.
  [14] Jerry and its remnants brought flash flooding to portions of the Upland South, particularly in the states of Kentucky, Virginia, and West Virginia.
  [15] In eastern Kentucky, hundreds of homes were flooded and many bridges, culverts, and roads were washed out; this left hundreds of residents stranded.
  [16] Damage in Kentucky reached at least $5 million.
  [17] Similar impact occurred in Virginia; with $3.4 million (1989 USD) in damage in Buchanan County.
  [18] In West Virginia, overflowing rivers in the western portions of the state forced hundreds to evacuate.
  [19] Throughout the United States, Jerry resulted in about $70 million in damage.

### Events (37 total)
  - EVENT_70824ec59e32d4f9f1edc2ba7e80ab90 | type=Motion | trigger='fell' | sent_id=12 | offset=[33, 34]
  - EVENT_6f6b8673d098cd0aaa97bceeb302674c | type=Coming_to_be | trigger='occurred' | sent_id=17 | offset=[2, 3]
  - EVENT_86f3e66fd22e185f6c132d1c197ebd31 | type=Causation | trigger='caused' | sent_id=12 | offset=[15, 16]
  - EVENT_cf4b81a7ad9f39307502eb49834092f7 | type=Damaging | trigger='Damage' | sent_id=16 | offset=[0, 1]
  - EVENT_365c60e7cad228a99ea281ea5131651c | type=Placing | trigger='situated' | sent_id=8 | offset=[13, 14]
  - EVENT_9f6f6fe534557ce6cfae4e1112bd7e9b | type=Manufacturing | trigger='made' | sent_id=6 | offset=[6, 7]
  - EVENT_f444a52cf6695b576d0b7605584dfaee | type=Motion | trigger='flooding' | sent_id=14 | offset=[6, 7]
  - EVENT_ff67303f3677d7116ea4b12642a7d19d | type=Name_conferral | trigger='named' | sent_id=1 | offset=[6, 7]
  - EVENT_4781ef6204d0b0f193bf43751e7b7fa3 | type=Damaging | trigger='damage' | sent_id=17 | offset=[15, 16]
  - EVENT_a2d921171a38375bb9144633ba93aea5 | type=Becoming | trigger='became' | sent_id=5 | offset=[9, 10]
  - EVENT_ed810b3e9d06f336f758294e203bc270 | type=Destroying | trigger='destroyed' | sent_id=9 | offset=[10, 11]
  - EVENT_699b3e589ac75144d73da86433379ba0 | type=Motion | trigger='flooded' | sent_id=15 | offset=[8, 9]
  - EVENT_15c35946717314c100eb593e8b17296b | type=Cause_change_of_strength | trigger='strengthened' | sent_id=2 | offset=[15, 16]
  - EVENT_b8feb2ec65264ea7a68ea9ed409109d0 | type=Arriving | trigger='reached' | sent_id=16 | offset=[3, 4]
  - EVENT_4653f332dcb148cb303d6ce788fc937f | type=Coming_to_be | trigger='developed' | sent_id=1 | offset=[13, 14]
  - EVENT_7bdd1c3844bb9973be36dc041c39bf5b | type=Motion_directional | trigger='curving' | sent_id=3 | offset=[11, 12]
  - EVENT_ec3f39f4a498a689e8ed65f234294885 | type=Preserving | trigger='maintained' | sent_id=3 | offset=[8, 9]
  - EVENT_b9f53da657d2a93134eba358a5ed226e | type=Motion | trigger='moved' | sent_id=2 | offset=[7, 8]
  - EVENT_5eff42ff0b51e23447968655bb2bbdcc | type=Cause_change_of_strength | trigger='deepened' | sent_id=3 | offset=[2, 3]
  - EVENT_020a452a1ec940cb9b9130763e1bd0a6 | type=Motion | trigger='re-curved' | sent_id=4 | offset=[6, 7]
  - EVENT_2ae8b11b9c00c4ae10a793356d1795c3 | type=Damaging | trigger='damage' | sent_id=11 | offset=[9, 10]
  - EVENT_002a08eb2d45cb5b60511d2b1e20772b | type=Reporting | trigger='reported' | sent_id=13 | offset=[7, 8]
  - EVENT_9351f3c28935762270e815a8b47cab34 | type=Bringing | trigger='brought' | sent_id=14 | offset=[4, 5]
  - EVENT_6c80346e5bf3be10bed651cce9e31ebb | type=Causation | trigger='resulted in' | sent_id=19 | offset=[6, 8]
  - EVENT_a95183aadc0236e47ab433d455527c3a | type=Causation | trigger='caused' | sent_id=0 | offset=[2, 3]
  - EVENT_a4a2d243d5b0fc507b4582c3fcde9bc5 | type=Killing | trigger='killing' | sent_id=12 | offset=[39, 40]
  - EVENT_904c65952df2b2b276e33678dc74ba80 | type=Coming_to_be | trigger='spawned' | sent_id=11 | offset=[17, 18]
  - EVENT_89d95681f9a7b2006ebba765923b5c0f | type=Damaging | trigger='damage' | sent_id=0 | offset=[4, 5]
  - EVENT_0e37d73c9901e6506ea3b779b206cdc7 | type=Motion | trigger='flooding' | sent_id=0 | offset=[9, 10]
  - EVENT_f0234afc666fc2eca2821424384998f3 | type=Process_start | trigger='began' | sent_id=5 | offset=[1, 2]

### causal_relations (2 entries)
  ** CAUSE: 8 pairs
    -> ['EVENT_f444a52cf6695b576d0b7605584dfaee', 'EVENT_cf4b81a7ad9f39307502eb49834092f7']
    -> ['EVENT_f444a52cf6695b576d0b7605584dfaee', 'EVENT_77441b0e18d134b8584dc50a20c76fd8']
    -> ['EVENT_0e37d73c9901e6506ea3b779b206cdc7', 'EVENT_cf4b81a7ad9f39307502eb49834092f7']
    -> ['EVENT_f444a52cf6695b576d0b7605584dfaee', 'EVENT_4781ef6204d0b0f193bf43751e7b7fa3']
    -> ['EVENT_904c65952df2b2b276e33678dc74ba80', 'EVENT_2ae8b11b9c00c4ae10a793356d1795c3']
    -> ['EVENT_0e37d73c9901e6506ea3b779b206cdc7', 'EVENT_77441b0e18d134b8584dc50a20c76fd8']
    -> ['EVENT_70824ec59e32d4f9f1edc2ba7e80ab90', 'EVENT_a4a2d243d5b0fc507b4582c3fcde9bc5']
    -> ['EVENT_f444a52cf6695b576d0b7605584dfaee', 'EVENT_699b3e589ac75144d73da86433379ba0']
  ** PRECONDITION: 3 pairs
    -> ['EVENT_0e37d73c9901e6506ea3b779b206cdc7', 'EVENT_6f9ed94d2dc419c31a15f23a5096cce3']
    -> ['EVENT_f0234afc666fc2eca2821424384998f3', 'EVENT_a2d921171a38375bb9144633ba93aea5']
    -> ['EVENT_f444a52cf6695b576d0b7605584dfaee', 'EVENT_6f9ed94d2dc419c31a15f23a5096cce3']
  ... (showing 10 of 11)

### temporal_relations (6 entries)
  ** BEFORE: 486 pairs
    -> ['TIME_d33ce93f453ddf764f6c204e3d7d8cef', 'EVENT_ed810b3e9d06f336f758294e203bc270']
    -> ['EVENT_0e37d73c9901e6506ea3b779b206cdc7', 'EVENT_cf4b81a7ad9f39307502eb49834092f7']
    -> ['TIME_e17dafb3b87e382a9fdf92a6abf53dee', 'EVENT_4781ef6204d0b0f193bf43751e7b7fa3']
    -> ['EVENT_9f6f6fe534557ce6cfae4e1112bd7e9b', 'TIME_d33ce93f453ddf764f6c204e3d7d8cef']
    -> ['EVENT_ed810b3e9d06f336f758294e203bc270', 'EVENT_f444a52cf6695b576d0b7605584dfaee']
    -> ['EVENT_89d95681f9a7b2006ebba765923b5c0f', 'EVENT_2ae8b11b9c00c4ae10a793356d1795c3']
    -> ['TIME_03b7f4455fbea229ce1dfd5362951fac', 'EVENT_a2d921171a38375bb9144633ba93aea5']
    -> ['EVENT_002a08eb2d45cb5b60511d2b1e20772b', 'EVENT_6f9ed94d2dc419c31a15f23a5096cce3']
    -> ['EVENT_5eff42ff0b51e23447968655bb2bbdcc', 'EVENT_2716b6b1567ec94488bae094eb9d2d21']
    -> ['EVENT_b9f53da657d2a93134eba358a5ed226e', 'EVENT_f444a52cf6695b576d0b7605584dfaee']
  ** OVERLAP: 2 pairs
    -> ['EVENT_89d95681f9a7b2006ebba765923b5c0f', 'EVENT_0e37d73c9901e6506ea3b779b206cdc7']
    -> ['EVENT_699b3e589ac75144d73da86433379ba0', 'EVENT_77441b0e18d134b8584dc50a20c76fd8']
  ** CONTAINS: 68 pairs
    -> ['TIME_d224a52135951d898ba5281b6d427302', 'EVENT_2716b6b1567ec94488bae094eb9d2d21']
    -> ['TIME_178172b7abeec20195e2faded4d1d1fa', 'EVENT_4653f332dcb148cb303d6ce788fc937f']
    -> ['EVENT_d4ccbe9065b78e76efe7fb2106cc2d5e', 'EVENT_b8feb2ec65264ea7a68ea9ed409109d0']
    -> ['TIME_178172b7abeec20195e2faded4d1d1fa', 'TIME_d33ce93f453ddf764f6c204e3d7d8cef']
    -> ['EVENT_d4ccbe9065b78e76efe7fb2106cc2d5e', 'TIME_03b7f4455fbea229ce1dfd5362951fac']
    -> ['TIME_178172b7abeec20195e2faded4d1d1fa', 'EVENT_2716b6b1567ec94488bae094eb9d2d21']
    -> ['EVENT_d4ccbe9065b78e76efe7fb2106cc2d5e', 'EVENT_699b3e589ac75144d73da86433379ba0']
    -> ['EVENT_d4ccbe9065b78e76efe7fb2106cc2d5e', 'EVENT_89d95681f9a7b2006ebba765923b5c0f']
    -> ['EVENT_d4ccbe9065b78e76efe7fb2106cc2d5e', 'EVENT_f0234afc666fc2eca2821424384998f3']
    -> ['TIME_a6ec0d543d5b74430ac63a5b0b4d7cf5', 'EVENT_70824ec59e32d4f9f1edc2ba7e80ab90']
  ** SIMULTANEOUS: 3 pairs
    -> ['EVENT_9351f3c28935762270e815a8b47cab34', 'EVENT_f444a52cf6695b576d0b7605584dfaee']
    -> ['EVENT_b8feb2ec65264ea7a68ea9ed409109d0', 'EVENT_cf4b81a7ad9f39307502eb49834092f7']
    -> ['EVENT_d4ccbe9065b78e76efe7fb2106cc2d5e', 'TIME_178172b7abeec20195e2faded4d1d1fa']
  ** ENDS-ON: 0 pairs
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 559)

---

## Sample 4: Death of Christopher Alder
**Doc ID**: ab70a3e49966caa8c35f8b27fabea3ad

### Sentences (15 total)
  [0] Christopher Alder was a trainee computer programmer and former British Army paratrooper who had served in the Falklands War and was commended for his service with the Army in Northern Ireland.
  [1] He died while in police custody at Queen's Gardens Police Station, Kingston upon Hull, in April 1998.
  [2] The case became a cause célèbre for civil rights campaigners in the United Kingdom.
  [3] He had earlier been the victim of an assault outside a nightclub and was taken to Hull Royal Infirmary where, possibly as a result of his head injury, staff said his behaviour was "extremely troubleso
  [4] He was escorted from the hospital by two police officers who arrested him to prevent a breach of the peace.
  [5] On arrival at the police station Alder was "partially dragged and partially carried," handcuffed and unconscious, from a police van and placed on the floor of the custody suite.
  [6] Officers chatted between themselves and speculated that he was faking illness.
  [7] Twelve minutes later one of the officers present noticed that Alder was not making any breathing noises and although resuscitation was attempted, he was pronounced dead at the scene.
  [8] A post mortem indicated that the head injury alone would not have killed him.
  [9] The incident was captured on the police station's closed-circuit television (CCTV) cameras.
  [10] A coroner's jury in 2000 returned a verdict that Alder was unlawfully killed.
  [11] In 2002 five police officers went on trial charged with Alder's manslaughter and misconduct in public office, but were acquitted on the orders of the judge.
  [12] In 2006 an Independent Police Complaints Commission report concluded that four of the officers present in the custody suite when Alder died were guilty of the "most serious neglect of duty".
  [13] In November 2011 the government formally apologised to Alder's family in the European Court of Human Rights, admitting that it had breached its obligations with regard to "preserving life and ensuring
  [14] They also admitted that they had failed to carry out an effective and independent inquiry into the case.

### Events (40 total)
  - EVENT_3ccd4f275de7691f301ab49019a1f7c5 | type=Preserving | trigger='preserving' | sent_id=13 | offset=[30, 31]
  - EVENT_2f8aae6529bb71a82b60c8da268eb782 | type=Reveal_secret | trigger='admitted' | sent_id=14 | offset=[2, 3]
  - EVENT_024c938b9f4dbe19c5549dbfc92cb328 | type=Patrolling | trigger='escorted' | sent_id=4 | offset=[2, 3]
  - EVENT_c11a88376fbbc0ec72a90da72845c09f | type=Carry_goods | trigger='carried' | sent_id=5 | offset=[13, 14]
  - EVENT_38fced5f4ff38829319f42024cbb239b | type=Attack | trigger='assault' | sent_id=3 | offset=[8, 9]
  - EVENT_f554f9b20e32e2d478835fb51a0fc24c | type=Know | trigger='noticed' | sent_id=7 | offset=[8, 9]
  - EVENT_529fb1a94861f0ec0205972716ef08e1 | type=Criminal_investigation | trigger='inquiry' | sent_id=14 | offset=[14, 15]
  - EVENT_f484a1f93f74f5bab552a236f111aa25 | type=Committing_crime | trigger='guilty' | sent_id=12 | offset=[23, 24]
  - EVENT_b6972d571bb31ae66692b8d08cf84592 | type=Killing | trigger='killed' | sent_id=10 | offset=[13, 14]
  - EVENT_d9ad5f5f0adc357a7ed104af3e4d5e82 | type=Killing | trigger='killed' | sent_id=8 | offset=[12, 13]
  - EVENT_04ea9f8fb521dc07781b0b95eea590a0 | type=Conquering | trigger='captured' | sent_id=9 | offset=[3, 4]
  - EVENT_003ef8bd2ededc5689aba7bc917740fd | type=Judgment_communication | trigger='commended' | sent_id=0 | offset=[21, 22]
  - EVENT_0c9465f5faa73cbaf7c49e575075efd8 | type=Coming_to_believe | trigger='concluded' | sent_id=12 | offset=[8, 9]
  - EVENT_1051a28585aa286e63d67cd7feec42fb | type=Legal_rulings | trigger='acquitted' | sent_id=11 | offset=[21, 22]
  - EVENT_f7011bdebe5301a29db5b24fdb0e9246 | type=Death | trigger='died' | sent_id=1 | offset=[1, 2]
  - EVENT_f7011bdebe5301a29db5b24fdb0e9246 | type=Death | trigger='died' | sent_id=12 | offset=[21, 22]
  - EVENT_6e9727748c0496cd62d1e2db26cc84d2 | type=Rescuing | trigger='resuscitation' | sent_id=7 | offset=[19, 20]
  - EVENT_93bfe939c4a9b98df2fc0cb725b9e575 | type=Bodily_harm | trigger='victim' | sent_id=3 | offset=[5, 6]
  - EVENT_de257bbae07bc0afadd99de8bf7432d0 | type=Judgment_communication | trigger='charged' | sent_id=11 | offset=[8, 9]
  - EVENT_e003a1d0d699feaf5967873077e51a9e | type=Legal_rulings | trigger='verdict' | sent_id=10 | offset=[8, 9]
  - EVENT_6f871436fb209292ac9385e0487f78da | type=Statement | trigger='pronounced' | sent_id=7 | offset=[25, 26]
  - EVENT_60986e24389de9cf1a8845358ea6ce8c | type=Check | trigger='indicated' | sent_id=8 | offset=[3, 4]
  - EVENT_647f7e71683ef5923fde982413d0247f | type=Criminal_investigation | trigger='trial' | sent_id=11 | offset=[7, 8]
  - EVENT_cdc173f12c3b1e81dd25d9c5f360591d | type=Judgment_communication | trigger='apologised' | sent_id=13 | offset=[6, 7]
  - EVENT_3c0e979d917825a74a0f88131fe08e38 | type=Telling | trigger='said' | sent_id=3 | offset=[31, 32]
  - EVENT_e8db867c1c2e3c70194be80cac8e00ef | type=Coming_to_believe | trigger='speculated' | sent_id=6 | offset=[5, 6]
  - EVENT_beba4975d4410d5b6116a4e41cf71637 | type=Preventing_or_letting | trigger='prevent' | sent_id=4 | offset=[14, 15]
  - EVENT_a15bdc9e0a8bacd31348900d5bf0cd5d | type=Communication | trigger='chatted' | sent_id=6 | offset=[1, 2]
  - EVENT_c25b95345ca040d0480046bacd9a94b7 | type=Bodily_harm | trigger='injury' | sent_id=3 | offset=[28, 29]
  - EVENT_c25b95345ca040d0480046bacd9a94b7 | type=Bodily_harm | trigger='injury' | sent_id=8 | offset=[7, 8]
  - EVENT_0217fe41b498e4b234fa262056883a83 | type=Arrest | trigger='arrested' | sent_id=4 | offset=[11, 12]
  - EVENT_ec8d68dd8ae6ea67eb659a16d0ee1f66 | type=Breathing | trigger='breathing' | sent_id=7 | offset=[15, 16]

### causal_relations (2 entries)
  ** CAUSE: 1 pairs
    -> ['EVENT_38fced5f4ff38829319f42024cbb239b', 'EVENT_93bfe939c4a9b98df2fc0cb725b9e575']
  ** PRECONDITION: 14 pairs
    -> ['EVENT_0217fe41b498e4b234fa262056883a83', 'EVENT_beba4975d4410d5b6116a4e41cf71637']
    -> ['EVENT_0217fe41b498e4b234fa262056883a83', 'EVENT_024c938b9f4dbe19c5549dbfc92cb328']
    -> ['EVENT_e003a1d0d699feaf5967873077e51a9e', 'EVENT_647f7e71683ef5923fde982413d0247f']
    -> ['EVENT_ec8d68dd8ae6ea67eb659a16d0ee1f66', 'EVENT_6e9727748c0496cd62d1e2db26cc84d2']
    -> ['EVENT_ec8d68dd8ae6ea67eb659a16d0ee1f66', 'EVENT_21a9cb44aad0d5b919587bbe83a00370']
    -> ['EVENT_de257bbae07bc0afadd99de8bf7432d0', 'EVENT_1051a28585aa286e63d67cd7feec42fb']
    -> ['EVENT_ec8d68dd8ae6ea67eb659a16d0ee1f66', 'EVENT_60986e24389de9cf1a8845358ea6ce8c']
    -> ['EVENT_a15bdc9e0a8bacd31348900d5bf0cd5d', 'EVENT_cdc173f12c3b1e81dd25d9c5f360591d']
    -> ['EVENT_38fced5f4ff38829319f42024cbb239b', 'EVENT_f3d5c7783dc83aaf34d314f712ed71ba']
    -> ['EVENT_f484a1f93f74f5bab552a236f111aa25', 'EVENT_cdc173f12c3b1e81dd25d9c5f360591d']
  ... (showing 10 of 15)

### temporal_relations (6 entries)
  ** BEFORE: 796 pairs
    -> ['EVENT_60986e24389de9cf1a8845358ea6ce8c', 'TIME_245b0d6508ada1a288a1585206afe458']
    -> ['EVENT_38fced5f4ff38829319f42024cbb239b', 'EVENT_0c9465f5faa73cbaf7c49e575075efd8']
    -> ['EVENT_6e9727748c0496cd62d1e2db26cc84d2', 'EVENT_e003a1d0d699feaf5967873077e51a9e']
    -> ['TIME_6dce0ffd27e9326e711b24bef029382a', 'EVENT_3ccd4f275de7691f301ab49019a1f7c5']
    -> ['EVENT_f3d5c7783dc83aaf34d314f712ed71ba', 'EVENT_0e07302936d3022ae97f85a6a438fcac']
    -> ['EVENT_a15bdc9e0a8bacd31348900d5bf0cd5d', 'EVENT_bd69c63af9c6afe27f01c2d1da13aa9b']
    -> ['TIME_e356ab699f6252b53930ead4ac6f296e', 'EVENT_6f871436fb209292ac9385e0487f78da']
    -> ['EVENT_f3d5c7783dc83aaf34d314f712ed71ba', 'EVENT_c3934c197f9d1bc85fe17cd35dea63fd']
    -> ['EVENT_f3d5c7783dc83aaf34d314f712ed71ba', 'EVENT_21a9cb44aad0d5b919587bbe83a00370']
    -> ['EVENT_e8db867c1c2e3c70194be80cac8e00ef', 'EVENT_2f8aae6529bb71a82b60c8da268eb782']
  ** OVERLAP: 1 pairs
    -> ['EVENT_a15bdc9e0a8bacd31348900d5bf0cd5d', 'EVENT_e8db867c1c2e3c70194be80cac8e00ef']
  ** CONTAINS: 49 pairs
    -> ['TIME_c4630b85e20c70b6e70a457abab0d510', 'EVENT_f3d5c7783dc83aaf34d314f712ed71ba']
    -> ['TIME_b6fcff1db8abd673d982fe0d8fad2dcd', 'EVENT_cdc173f12c3b1e81dd25d9c5f360591d']
    -> ['TIME_c4630b85e20c70b6e70a457abab0d510', 'EVENT_0217fe41b498e4b234fa262056883a83']
    -> ['EVENT_04ea9f8fb521dc07781b0b95eea590a0', 'EVENT_e8db867c1c2e3c70194be80cac8e00ef']
    -> ['EVENT_04ea9f8fb521dc07781b0b95eea590a0', 'EVENT_6e9727748c0496cd62d1e2db26cc84d2']
    -> ['EVENT_04ea9f8fb521dc07781b0b95eea590a0', 'EVENT_6f871436fb209292ac9385e0487f78da']
    -> ['TIME_245b0d6508ada1a288a1585206afe458', 'EVENT_e003a1d0d699feaf5967873077e51a9e']
    -> ['TIME_c4630b85e20c70b6e70a457abab0d510', 'EVENT_3c0e979d917825a74a0f88131fe08e38']
    -> ['TIME_b6fcff1db8abd673d982fe0d8fad2dcd', 'EVENT_3ccd4f275de7691f301ab49019a1f7c5']
    -> ['EVENT_04ea9f8fb521dc07781b0b95eea590a0', 'EVENT_c11a88376fbbc0ec72a90da72845c09f']
  ** SIMULTANEOUS: 4 pairs
    -> ['EVENT_e003a1d0d699feaf5967873077e51a9e', 'EVENT_bd69c63af9c6afe27f01c2d1da13aa9b']
    -> ['EVENT_6f871436fb209292ac9385e0487f78da', 'EVENT_21a9cb44aad0d5b919587bbe83a00370']
    -> ['EVENT_647f7e71683ef5923fde982413d0247f', 'EVENT_de257bbae07bc0afadd99de8bf7432d0']
    -> ['EVENT_c3934c197f9d1bc85fe17cd35dea63fd', 'EVENT_c11a88376fbbc0ec72a90da72845c09f']
  ** ENDS-ON: 0 pairs
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 850)

---

## Sample 5: Clackamas Town Center shooting
**Doc ID**: 49e16a75dfd748513d5db9f8a849d218

### Sentences (6 total)
  [0] On December 11, 2012, a shooting occurred at the Clackamas Town Center in unincorporated Clackamas County, outside the city of Portland, Oregon, United States.
  [1] The gunman, 22-year-old Jacob Tyler Roberts, ran into the shopping center wearing tactical clothing and a hockey mask and opened fire on shoppers and employees with a stolen Stag Arms AR-15 rifle.
  [2] He fired a total of 17 shots, killing two people and seriously wounding a third.
  [3] Having attempted to reload his weapon and dropping three magazines, Roberts entered a stairwell and committed suicide after descending one level because he encountered someone with a concealed carry p
  [4] He had no connection to any of his victims, and it was believed to be a random act of violence.
  [5] The Clackamas Town Center has a posted policy of prohibiting firearms on the premises.

### Events (17 total)
  - EVENT_422ed35102bdb680bc098aa3a94a724b | type=Killing | trigger='suicide' | sent_id=3 | offset=[17, 18]
  - EVENT_9590eefd428bcee365ddf8df71f6294b | type=Self_motion | trigger='ran' | sent_id=1 | offset=[8, 9]
  - EVENT_467a65124e06a33e7b8b37f6848f14ef | type=Committing_crime | trigger='committed' | sent_id=3 | offset=[16, 17]
  - EVENT_36fb836332007359ad085f8f6f6d7f48 | type=Coming_to_be | trigger='occurred' | sent_id=0 | offset=[8, 9]
  - EVENT_697a06d1d172abef670f3dfd3c353862 | type=Motion | trigger='drawn' | sent_id=3 | offset=[33, 34]
  - EVENT_7fa8ef0a48f0d82a7da1c060077e571a | type=Motion_directional | trigger='descending' | sent_id=3 | offset=[19, 20]
  - EVENT_2ba609b3aacac38e6c2b84e009a1cd90 | type=Use_firearm | trigger='fired' | sent_id=2 | offset=[1, 2]
  - EVENT_413c88546068f1e94c400eccd202a987 | type=Use_firearm | trigger='fire' | sent_id=1 | offset=[22, 23]
  - EVENT_fd971030f35f067fe1a616818b5fb4b8 | type=Body_movement | trigger='dropping' | sent_id=3 | offset=[7, 8]
  - EVENT_ce4bef9f5ba9605323ca2c33ccf4c81e | type=Use_firearm | trigger='shooting' | sent_id=0 | offset=[7, 8]
  - EVENT_93afd9d004189af595b6a9e514b7fe57 | type=Preventing_or_letting | trigger='prohibiting' | sent_id=5 | offset=[9, 10]
  - EVENT_23d17d77282e3a2639abb03e7ed68649 | type=Killing | trigger='killing' | sent_id=2 | offset=[8, 9]
  - EVENT_d360c082c167942f763e90550f9f6baf | type=Bodily_harm | trigger='wounding' | sent_id=2 | offset=[13, 14]
  - EVENT_d31237a3dc4e16865b2147afd03e444f | type=Know | trigger='believed' | sent_id=4 | offset=[13, 14]
  - EVENT_38dc7006c11a2ba97036ba4f849026eb | type=Wearing | trigger='wearing' | sent_id=1 | offset=[13, 14]
  - EVENT_9bc177cfd64c67c8c9f1cbcb90b5582a | type=Arriving | trigger='entered' | sent_id=3 | offset=[12, 13]
  - EVENT_d210b4e2545143c8e26e40f9ebd15f29 | type=Violence | trigger='violence' | sent_id=4 | offset=[20, 21]

### causal_relations (2 entries)
  ** CAUSE: 2 pairs
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_d360c082c167942f763e90550f9f6baf']
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_23d17d77282e3a2639abb03e7ed68649']
  ** PRECONDITION: 1 pairs
    -> ['EVENT_7fa8ef0a48f0d82a7da1c060077e571a', 'EVENT_422ed35102bdb680bc098aa3a94a724b']

### temporal_relations (6 entries)
  ** BEFORE: 67 pairs
    -> ['EVENT_9590eefd428bcee365ddf8df71f6294b', 'EVENT_413c88546068f1e94c400eccd202a987']
    -> ['EVENT_d210b4e2545143c8e26e40f9ebd15f29', 'EVENT_467a65124e06a33e7b8b37f6848f14ef']
    -> ['EVENT_7fa8ef0a48f0d82a7da1c060077e571a', 'EVENT_422ed35102bdb680bc098aa3a94a724b']
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_7fa8ef0a48f0d82a7da1c060077e571a']
    -> ['EVENT_fd971030f35f067fe1a616818b5fb4b8', 'EVENT_422ed35102bdb680bc098aa3a94a724b']
    -> ['EVENT_9590eefd428bcee365ddf8df71f6294b', 'EVENT_422ed35102bdb680bc098aa3a94a724b']
    -> ['EVENT_ce4bef9f5ba9605323ca2c33ccf4c81e', 'EVENT_fd971030f35f067fe1a616818b5fb4b8']
    -> ['EVENT_36fb836332007359ad085f8f6f6d7f48', 'EVENT_93afd9d004189af595b6a9e514b7fe57']
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_9bc177cfd64c67c8c9f1cbcb90b5582a']
    -> ['EVENT_697a06d1d172abef670f3dfd3c353862', 'EVENT_422ed35102bdb680bc098aa3a94a724b']
  ** OVERLAP: 13 pairs
    -> ['EVENT_ce4bef9f5ba9605323ca2c33ccf4c81e', 'EVENT_23d17d77282e3a2639abb03e7ed68649']
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_23d17d77282e3a2639abb03e7ed68649']
    -> ['TIME_26d49779da7be1d48e35f72cbfedb016', 'EVENT_93afd9d004189af595b6a9e514b7fe57']
    -> ['EVENT_413c88546068f1e94c400eccd202a987', 'EVENT_23d17d77282e3a2639abb03e7ed68649']
    -> ['EVENT_d210b4e2545143c8e26e40f9ebd15f29', 'EVENT_23d17d77282e3a2639abb03e7ed68649']
    -> ['EVENT_38dc7006c11a2ba97036ba4f849026eb', 'EVENT_93afd9d004189af595b6a9e514b7fe57']
    -> ['EVENT_36fb836332007359ad085f8f6f6d7f48', 'EVENT_d360c082c167942f763e90550f9f6baf']
    -> ['EVENT_23d17d77282e3a2639abb03e7ed68649', 'EVENT_d360c082c167942f763e90550f9f6baf']
    -> ['EVENT_ce4bef9f5ba9605323ca2c33ccf4c81e', 'EVENT_d360c082c167942f763e90550f9f6baf']
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_d360c082c167942f763e90550f9f6baf']
  ** CONTAINS: 25 pairs
    -> ['EVENT_38dc7006c11a2ba97036ba4f849026eb', 'EVENT_2ba609b3aacac38e6c2b84e009a1cd90']
    -> ['TIME_26d49779da7be1d48e35f72cbfedb016', 'EVENT_fd971030f35f067fe1a616818b5fb4b8']
    -> ['TIME_26d49779da7be1d48e35f72cbfedb016', 'EVENT_413c88546068f1e94c400eccd202a987']
    -> ['EVENT_38dc7006c11a2ba97036ba4f849026eb', 'EVENT_697a06d1d172abef670f3dfd3c353862']
    -> ['TIME_26d49779da7be1d48e35f72cbfedb016', 'EVENT_422ed35102bdb680bc098aa3a94a724b']
    -> ['EVENT_38dc7006c11a2ba97036ba4f849026eb', 'EVENT_fd971030f35f067fe1a616818b5fb4b8']
    -> ['TIME_26d49779da7be1d48e35f72cbfedb016', 'EVENT_d210b4e2545143c8e26e40f9ebd15f29']
    -> ['TIME_26d49779da7be1d48e35f72cbfedb016', 'EVENT_ce4bef9f5ba9605323ca2c33ccf4c81e']
    -> ['TIME_26d49779da7be1d48e35f72cbfedb016', 'EVENT_38dc7006c11a2ba97036ba4f849026eb']
    -> ['TIME_26d49779da7be1d48e35f72cbfedb016', 'EVENT_7fa8ef0a48f0d82a7da1c060077e571a']
  ** SIMULTANEOUS: 11 pairs
    -> ['EVENT_422ed35102bdb680bc098aa3a94a724b', 'EVENT_467a65124e06a33e7b8b37f6848f14ef']
    -> ['EVENT_d210b4e2545143c8e26e40f9ebd15f29', 'EVENT_36fb836332007359ad085f8f6f6d7f48']
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_36fb836332007359ad085f8f6f6d7f48']
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_d210b4e2545143c8e26e40f9ebd15f29']
    -> ['EVENT_413c88546068f1e94c400eccd202a987', 'EVENT_ce4bef9f5ba9605323ca2c33ccf4c81e']
    -> ['EVENT_d210b4e2545143c8e26e40f9ebd15f29', 'EVENT_ce4bef9f5ba9605323ca2c33ccf4c81e']
    -> ['EVENT_413c88546068f1e94c400eccd202a987', 'EVENT_36fb836332007359ad085f8f6f6d7f48']
    -> ['EVENT_413c88546068f1e94c400eccd202a987', 'EVENT_d210b4e2545143c8e26e40f9ebd15f29']
    -> ['EVENT_ce4bef9f5ba9605323ca2c33ccf4c81e', 'EVENT_36fb836332007359ad085f8f6f6d7f48']
    -> ['EVENT_2ba609b3aacac38e6c2b84e009a1cd90', 'EVENT_413c88546068f1e94c400eccd202a987']
  ** ENDS-ON: 0 pairs
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 116)

---

## Sample 6: Más Para Dar
**Doc ID**: d90fc4fe526f0c06db854321a64f9c86

### Sentences (9 total)
  [0] Más Para Dar, is the twenty-seventh (27th) studio album by Puerto Rican singer Yolandita Monge and her first release in over four years.
  [1] This album was released on November 13, 2012.
  [2] It contains nine new songs co-written by Yolandita Monge, being the first time in the singer's career that she composes for an entire album.
  [3] This release follows the same musical and lyrical style as her previous studio albums Demasiado Fuerte and Mala and was produced once again by Jose Luis Pagán.
  [4] The album is a balanced effort, one that mixes pop and rock with the instant highlights "Ahora Vivo Si Tu Amor" and the electro-fueled single "Vivo Por Tí", which get a moving acoustic reading later i
  [5] "Desde Que Te Perdí" is both a slow builder and slow burner, with layers of guitars and strings supporting Monge's expressive and powerful voice.
  [6] "Verás Dolor" and "Y Aquí Me Ves De Pie" prove that the singer's emotional delivery is better than ever.
  [7] The title track "Más Para Dar" is a beautiful song about second chances in love and life.
  [8] This albums finds Monge going the indie route by releasing it in her own imprint, Roma Entertainment, and it is available as a digital download at iTunes and Amazon.

### Events (7 total)
  - EVENT_239ada452bafa37315235a834b451868 | type=Manufacturing | trigger='produced' | sent_id=3 | offset=[20, 21]
  - EVENT_4b5c800d8b55fbf22ab8fe1e48f12303 | type=Know | trigger='finds' | sent_id=8 | offset=[2, 3]
  - EVENT_e2ef17917af24c6727ff97dc998088d2 | type=Motion | trigger='going' | sent_id=8 | offset=[4, 5]
  - EVENT_db875761843098ec2523bf622c7b4dd5 | type=Come_together | trigger='composes' | sent_id=2 | offset=[21, 22]
  - EVENT_83b789041de9a493a5a7db0980159f5f | type=Publishing | trigger='release' | sent_id=0 | offset=[21, 22]
  - EVENT_83b789041de9a493a5a7db0980159f5f | type=Publishing | trigger='released' | sent_id=1 | offset=[3, 4]
  - EVENT_83b789041de9a493a5a7db0980159f5f | type=Publishing | trigger='release' | sent_id=3 | offset=[1, 2]
  - EVENT_83b789041de9a493a5a7db0980159f5f | type=Publishing | trigger='releasing' | sent_id=8 | offset=[9, 10]
  - EVENT_7d728912e221e787c3e7391135e707b0 | type=Supporting | trigger='supporting' | sent_id=5 | offset=[21, 22]
  - EVENT_431ba38e87085fa6fe51611142da6a4a | type=Getting | trigger='get' | sent_id=4 | offset=[35, 36]

### causal_relations (2 entries)
  ** CAUSE: 0 pairs
  ** PRECONDITION: 3 pairs
    -> ['EVENT_83b789041de9a493a5a7db0980159f5f', 'EVENT_e2ef17917af24c6727ff97dc998088d2']
    -> ['EVENT_4b5c800d8b55fbf22ab8fe1e48f12303', 'EVENT_e2ef17917af24c6727ff97dc998088d2']
    -> ['EVENT_83b789041de9a493a5a7db0980159f5f', 'EVENT_4b5c800d8b55fbf22ab8fe1e48f12303']

### temporal_relations (6 entries)
  ** BEFORE: 19 pairs
    -> ['TIME_0ffd4691f4a563ae2ddef71ebd3296ab', 'TIME_57c31fcf777db72ee4a8c2b555421d38']
    -> ['EVENT_83b789041de9a493a5a7db0980159f5f', 'TIME_57c31fcf777db72ee4a8c2b555421d38']
    -> ['EVENT_431ba38e87085fa6fe51611142da6a4a', 'EVENT_e2ef17917af24c6727ff97dc998088d2']
    -> ['EVENT_239ada452bafa37315235a834b451868', 'TIME_0ffd4691f4a563ae2ddef71ebd3296ab']
    -> ['EVENT_239ada452bafa37315235a834b451868', 'EVENT_4b5c800d8b55fbf22ab8fe1e48f12303']
    -> ['EVENT_431ba38e87085fa6fe51611142da6a4a', 'TIME_57c31fcf777db72ee4a8c2b555421d38']
    -> ['EVENT_239ada452bafa37315235a834b451868', 'EVENT_e2ef17917af24c6727ff97dc998088d2']
    -> ['EVENT_239ada452bafa37315235a834b451868', 'EVENT_431ba38e87085fa6fe51611142da6a4a']
    -> ['EVENT_431ba38e87085fa6fe51611142da6a4a', 'EVENT_83b789041de9a493a5a7db0980159f5f']
    -> ['TIME_0ffd4691f4a563ae2ddef71ebd3296ab', 'EVENT_4b5c800d8b55fbf22ab8fe1e48f12303']
  ** OVERLAP: 1 pairs
    -> ['EVENT_4b5c800d8b55fbf22ab8fe1e48f12303', 'EVENT_e2ef17917af24c6727ff97dc998088d2']
  ** CONTAINS: 1 pairs
    -> ['TIME_0ffd4691f4a563ae2ddef71ebd3296ab', 'EVENT_83b789041de9a493a5a7db0980159f5f']
  ** SIMULTANEOUS: 0 pairs
  ** ENDS-ON: 0 pairs
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 21)

---

## Sample 7: Assassination of Abraham Lincoln
**Doc ID**: e8b093a299287c3007e7cb1fa952f059

### Sentences (8 total)
  [0] Abraham Lincoln, the 16th President of the United States, was assassinated by well-known stage actor John Wilkes Booth on April 14, 1865, while attending the play "Our American Cousin" at Ford's Theat
  [1] Shot in the head as he watched the play, Lincoln died the following day at 7:22 am, in the Petersen House opposite the theater.
  [2] He was the first U.S. president to be assassinated, and Lincoln's funeral and burial marked an extended period of national mourning.
  [3] Occurring near the end of the American Civil War, the assassination was part of a larger conspiracy intended by Booth to revive the Confederate cause by eliminating the three most important officials 
  [4] Conspirators Lewis Powell and David Herold were assigned to kill Secretary of State William H. Seward, and George Atzerodt was tasked with killing Vice President Andrew Johnson.
  [5] Beyond Lincoln's death, the plot failed: Seward was only wounded and Johnson's would-be attacker lost his nerve.
  [6] After a dramatic initial escape, Booth was killed at the climax of a 12-day manhunt.
  [7] Powell, Herold, Atzerodt and Mary Surratt were later hanged for their roles in the conspiracy.

### Events (20 total)
  - EVENT_8b23ba811318084e32c79d1d2363af9b | type=Request | trigger='tasked' | sent_id=4 | offset=[21, 22]
  - EVENT_2ebfb0097165295734f09268ca7db83a | type=Killing | trigger='assassinated' | sent_id=0 | offset=[12, 13]
  - EVENT_2ebfb0097165295734f09268ca7db83a | type=Killing | trigger='assassinated' | sent_id=2 | offset=[8, 9]
  - EVENT_2ebfb0097165295734f09268ca7db83a | type=Killing | trigger='assassination' | sent_id=3 | offset=[11, 12]
  - EVENT_826dfaa51f4396b358268e3ff803cd11 | type=Perception_active | trigger='watched' | sent_id=1 | offset=[6, 7]
  - EVENT_f901d3fc2bd768c7259c5e9f98627fd4 | type=Rite | trigger='burial' | sent_id=2 | offset=[15, 16]
  - EVENT_f6226be2d8f030e4dd05b09827b6b8d3 | type=Bodily_harm | trigger='wounded' | sent_id=5 | offset=[12, 13]
  - EVENT_46d3fe6fb12cf84f181b9170cdfd88a5 | type=Rite | trigger='funeral' | sent_id=2 | offset=[13, 14]
  - EVENT_a73e573e2813e19678e58e1e6b376a78 | type=Participation | trigger='attending' | sent_id=0 | offset=[27, 28]
  - EVENT_592fab4d4b2d7d589b50a5c681beea1a | type=Participation | trigger='play' | sent_id=0 | offset=[29, 30]
  - EVENT_b5568e907c64dbdd99da1d2622a002eb | type=Killing | trigger='killed' | sent_id=6 | offset=[8, 9]
  - EVENT_5ac47510b2f4609188cce4a291e8f9b2 | type=Escaping | trigger='escape' | sent_id=6 | offset=[4, 5]
  - EVENT_da1114c7d677b2414e46ee09094427c8 | type=Death | trigger='death' | sent_id=5 | offset=[3, 4]
  - EVENT_29b95de51e1c048420fb837facf15284 | type=Earnings_and_losses | trigger='lost' | sent_id=5 | offset=[18, 19]
  - EVENT_c9e54335d59786528bf80462a6882101 | type=Killing | trigger='killing' | sent_id=4 | offset=[23, 24]
  - EVENT_9b1d2a3461714a2f2ef475e160178955 | type=Recovering | trigger='revive' | sent_id=3 | offset=[22, 23]
  - EVENT_6b1a5b8c29f82d71db2ff15d956eb42e | type=Killing | trigger='kill' | sent_id=4 | offset=[9, 10]
  - EVENT_40fa27a757ad2be57acb2c57e2e312c4 | type=Use_firearm | trigger='shot' | sent_id=1 | offset=[0, 1]
  - EVENT_c85e82bf92572c29c566a151b8e43d95 | type=Recording | trigger='marked' | sent_id=2 | offset=[16, 17]
  - EVENT_de683486255b8f45552fdb0f38da4d60 | type=Death | trigger='died' | sent_id=1 | offset=[11, 12]
  - EVENT_7615d847bd66e893dfacf56f348b9f45 | type=Military_operation | trigger='War' | sent_id=3 | offset=[8, 9]
  - EVENT_6fb791356f2e6e1791f3647321be129e | type=Request | trigger='assigned' | sent_id=4 | offset=[7, 8]

### causal_relations (2 entries)
  ** CAUSE: 3 pairs
    -> ['EVENT_40fa27a757ad2be57acb2c57e2e312c4', 'EVENT_de683486255b8f45552fdb0f38da4d60']
    -> ['EVENT_46d3fe6fb12cf84f181b9170cdfd88a5', 'EVENT_c85e82bf92572c29c566a151b8e43d95']
    -> ['EVENT_f901d3fc2bd768c7259c5e9f98627fd4', 'EVENT_c85e82bf92572c29c566a151b8e43d95']
  ** PRECONDITION: 12 pairs
    -> ['EVENT_40fa27a757ad2be57acb2c57e2e312c4', 'EVENT_c85e82bf92572c29c566a151b8e43d95']
    -> ['EVENT_6fb791356f2e6e1791f3647321be129e', 'EVENT_f6226be2d8f030e4dd05b09827b6b8d3']
    -> ['EVENT_8b23ba811318084e32c79d1d2363af9b', 'EVENT_29b95de51e1c048420fb837facf15284']
    -> ['EVENT_2ebfb0097165295734f09268ca7db83a', 'EVENT_5ac47510b2f4609188cce4a291e8f9b2']
    -> ['EVENT_2ebfb0097165295734f09268ca7db83a', 'EVENT_b5568e907c64dbdd99da1d2622a002eb']
    -> ['EVENT_de683486255b8f45552fdb0f38da4d60', 'EVENT_c85e82bf92572c29c566a151b8e43d95']
    -> ['EVENT_8b23ba811318084e32c79d1d2363af9b', 'EVENT_c9e54335d59786528bf80462a6882101']
    -> ['EVENT_40fa27a757ad2be57acb2c57e2e312c4', 'EVENT_46d3fe6fb12cf84f181b9170cdfd88a5']
    -> ['EVENT_de683486255b8f45552fdb0f38da4d60', 'EVENT_46d3fe6fb12cf84f181b9170cdfd88a5']
    -> ['EVENT_de683486255b8f45552fdb0f38da4d60', 'EVENT_f901d3fc2bd768c7259c5e9f98627fd4']
  ... (showing 10 of 15)

### temporal_relations (6 entries)
  ** BEFORE: 107 pairs
    -> ['EVENT_8b23ba811318084e32c79d1d2363af9b', 'EVENT_f6226be2d8f030e4dd05b09827b6b8d3']
    -> ['TIME_da5cfa736bd1081916c0cb373b96de4f', 'EVENT_f901d3fc2bd768c7259c5e9f98627fd4']
    -> ['EVENT_f6226be2d8f030e4dd05b09827b6b8d3', 'EVENT_da1114c7d677b2414e46ee09094427c8']
    -> ['EVENT_da1114c7d677b2414e46ee09094427c8', 'EVENT_b5568e907c64dbdd99da1d2622a002eb']
    -> ['EVENT_de683486255b8f45552fdb0f38da4d60', 'EVENT_c85e82bf92572c29c566a151b8e43d95']
    -> ['EVENT_6fb791356f2e6e1791f3647321be129e', 'EVENT_c9e54335d59786528bf80462a6882101']
    -> ['EVENT_f6226be2d8f030e4dd05b09827b6b8d3', 'EVENT_b5568e907c64dbdd99da1d2622a002eb']
    -> ['TIME_e91a5d27d032970d1d56b3d470439075', 'EVENT_de683486255b8f45552fdb0f38da4d60']
    -> ['EVENT_de683486255b8f45552fdb0f38da4d60', 'EVENT_46d3fe6fb12cf84f181b9170cdfd88a5']
    -> ['EVENT_6fb791356f2e6e1791f3647321be129e', 'EVENT_f901d3fc2bd768c7259c5e9f98627fd4']
  ** OVERLAP: 1 pairs
    -> ['TIME_da5cfa736bd1081916c0cb373b96de4f', 'EVENT_5ac47510b2f4609188cce4a291e8f9b2']
  ** CONTAINS: 47 pairs
    -> ['TIME_1ba1507f61299984299c4888587f0147', 'EVENT_5ac47510b2f4609188cce4a291e8f9b2']
    -> ['TIME_da5cfa736bd1081916c0cb373b96de4f', 'EVENT_826dfaa51f4396b358268e3ff803cd11']
    -> ['EVENT_826dfaa51f4396b358268e3ff803cd11', 'EVENT_f6226be2d8f030e4dd05b09827b6b8d3']
    -> ['TIME_da5cfa736bd1081916c0cb373b96de4f', 'EVENT_40fa27a757ad2be57acb2c57e2e312c4']
    -> ['EVENT_826dfaa51f4396b358268e3ff803cd11', 'EVENT_29b95de51e1c048420fb837facf15284']
    -> ['EVENT_7615d847bd66e893dfacf56f348b9f45', 'EVENT_46d3fe6fb12cf84f181b9170cdfd88a5']
    -> ['EVENT_7615d847bd66e893dfacf56f348b9f45', 'EVENT_6fb791356f2e6e1791f3647321be129e']
    -> ['EVENT_7615d847bd66e893dfacf56f348b9f45', 'EVENT_8b23ba811318084e32c79d1d2363af9b']
    -> ['TIME_1ba1507f61299984299c4888587f0147', 'TIME_e91a5d27d032970d1d56b3d470439075']
    -> ['EVENT_5ac47510b2f4609188cce4a291e8f9b2', 'EVENT_da1114c7d677b2414e46ee09094427c8']
  ** SIMULTANEOUS: 0 pairs
  ** ENDS-ON: 0 pairs
  ** BEGINS-ON: 1 pairs
    -> ['TIME_da5cfa736bd1081916c0cb373b96de4f', 'TIME_1ba1507f61299984299c4888587f0147']
  ... (showing 10 of 156)

### subevent_relations (5 entries)
  -> ['EVENT_7615d847bd66e893dfacf56f348b9f45', 'EVENT_8b23ba811318084e32c79d1d2363af9b']
  -> ['EVENT_7615d847bd66e893dfacf56f348b9f45', 'EVENT_40fa27a757ad2be57acb2c57e2e312c4']
  -> ['EVENT_7615d847bd66e893dfacf56f348b9f45', 'EVENT_6fb791356f2e6e1791f3647321be129e']
  -> ['EVENT_2ebfb0097165295734f09268ca7db83a', 'EVENT_40fa27a757ad2be57acb2c57e2e312c4']
  -> ['EVENT_7615d847bd66e893dfacf56f348b9f45', 'EVENT_2ebfb0097165295734f09268ca7db83a']

---

## Sample 8: Battle of Rocquencourt
**Doc ID**: c20868c301a1b2170bacc32d02b25a32

### Sentences (9 total)
  [0] The Battle of Rocquencourt was a cavalry skirmish fought on 1 July 1815 in and around the villages of Rocquencourt and Le Chesnay.
  [1] French dragoons supported by infantry and commanded by General Exelmans destroyed a Prussian brigade of hussars under the command of Lieutenant Colonel Eston von Sohr (who was severely wounded and tak
  [2] Prussian cavalry detachment under the command of Lieutenant Colonel Sohr ventured too far in advance of the main body of the Prussian army with the intention of reaching the Orléans road from Paris; w
  [3] However, when the Prussian detachment was in the vicinity of Rocquencourt it was ambushed by a superior French force.
  [4] Under attack the Prussians retreated from Versailles and headed east, but were blocked by the French at Vélizy.
  [5] They failed to re-enter Versailles and headed for Saint-Germain-en-Laye.
  [6] Their first squadron came under fire at the entrance of Rocquencourt and attempted to escape through the fields.
  [7] They were forced into a small, narrow street in Le Chesnay and killed or captured.
  [8] Just before nightfall the same day, the advanced guard of the Prussian III Corps, having heard of the destruction of Sohr's detachment, succeeded in recapturing Rocquencourt and bivouacked there.

### Events (26 total)
  - EVENT_aca424608d0738b0f38c84ccd6a5730d | type=Conquering | trigger='recapturing' | sent_id=8 | offset=[28, 29]
  - EVENT_8ec16546e047fec7e22aa769ae267012 | type=Control | trigger='command' | sent_id=2 | offset=[5, 6]
  - EVENT_d2faf24fb4086f49cfef3e8014dd3781 | type=Supporting | trigger='supported' | sent_id=1 | offset=[2, 3]
  - EVENT_406a1f0aa7237ec3b9a9a65574986d5e | type=Manufacturing | trigger='produced' | sent_id=2 | offset=[50, 51]
  - EVENT_b0f5453c636e130e5185f688140a1cc4 | type=Cause_change_of_position_on_a_scale | trigger='increase' | sent_id=2 | offset=[46, 47]
  - EVENT_7dcd4c35a18eafcd79bdd8f2ed8890d6 | type=Self_motion | trigger='headed' | sent_id=5 | offset=[6, 7]
  - EVENT_b374c762a61a1b07126c71525d63a4c5 | type=Escaping | trigger='retreated' | sent_id=4 | offset=[4, 5]
  - EVENT_3c2d611ee315668a59e18c42c42d093c | type=Escaping | trigger='escape' | sent_id=6 | offset=[14, 15]
  - EVENT_f12af731e3f0ffbccf74b1d60af19c99 | type=Causation | trigger='forced' | sent_id=7 | offset=[2, 3]
  - EVENT_22c6e132281aa7754646b01cfae031af | type=Risk | trigger='ventured' | sent_id=2 | offset=[10, 11]
  - EVENT_2d90e40c703c2cb9fa3ad3d147113ef9 | type=Conquering | trigger='taken' | sent_id=1 | offset=[31, 32]
  - EVENT_d8c5e375163e6235d40b4260da764ae5 | type=Bodily_harm | trigger='wounded' | sent_id=1 | offset=[29, 30]
  - EVENT_1793cae7aaf55ac974a98b49664311b9 | type=Self_motion | trigger='headed' | sent_id=4 | offset=[8, 9]
  - EVENT_8e145a194b6c9259e961716524a7a4a9 | type=Attack | trigger='fire' | sent_id=6 | offset=[5, 6]
  - EVENT_483a01318abb41f4245d664ff0bd490c | type=Control | trigger='commanded' | sent_id=1 | offset=[6, 7]
  - EVENT_3de256733dc4c4c0db9fc61e158699b9 | type=Arriving | trigger='reaching' | sent_id=2 | offset=[27, 28]
  - EVENT_17406397a66051f826e8f3ced64ab6ac | type=Destroying | trigger='destruction' | sent_id=8 | offset=[20, 21]
  - EVENT_f7dc4b4002a9bb1857f9c5943a7fa697 | type=Hindering | trigger='blocked' | sent_id=4 | offset=[13, 14]
  - EVENT_763ad23a774723f74a9ac6ca430a8d45 | type=Conquering | trigger='captured' | sent_id=7 | offset=[15, 16]
  - EVENT_101ac6c53adcbbeead2630740356836d | type=Attack | trigger='ambushed' | sent_id=3 | offset=[14, 15]
  - EVENT_ca8e7a084029b8b7e02eb8842acb141f | type=Hostile_encounter | trigger='Battle' | sent_id=0 | offset=[1, 2]
  - EVENT_ca8e7a084029b8b7e02eb8842acb141f | type=Hostile_encounter | trigger='fought' | sent_id=0 | offset=[8, 9]
  - EVENT_253d785f0986f4a957b104b28a5984eb | type=Perception_active | trigger='heard' | sent_id=8 | offset=[17, 18]
  - EVENT_e65e2c3d3a42e514ad23446c912950bd | type=Control | trigger='command' | sent_id=1 | offset=[18, 19]
  - EVENT_854e2675ef627a5052293dbc38d9a94c | type=Self_motion | trigger='came' | sent_id=6 | offset=[3, 4]
  - EVENT_2f9194041946343afae4076b4a914512 | type=Destroying | trigger='destroyed' | sent_id=1 | offset=[10, 11]
  - EVENT_35de36460b07aa617fc730fb974fb329 | type=Killing | trigger='killed' | sent_id=7 | offset=[13, 14]

### causal_relations (2 entries)
  ** CAUSE: 0 pairs
  ** PRECONDITION: 26 pairs
    -> ['EVENT_ca8e7a084029b8b7e02eb8842acb141f', 'EVENT_2f9194041946343afae4076b4a914512']
    -> ['EVENT_17406397a66051f826e8f3ced64ab6ac', 'EVENT_aca424608d0738b0f38c84ccd6a5730d']
    -> ['EVENT_ca8e7a084029b8b7e02eb8842acb141f', 'EVENT_483a01318abb41f4245d664ff0bd490c']
    -> ['EVENT_854e2675ef627a5052293dbc38d9a94c', 'EVENT_763ad23a774723f74a9ac6ca430a8d45']
    -> ['EVENT_b374c762a61a1b07126c71525d63a4c5', 'EVENT_f7dc4b4002a9bb1857f9c5943a7fa697']
    -> ['EVENT_d2faf24fb4086f49cfef3e8014dd3781', 'EVENT_2f9194041946343afae4076b4a914512']
    -> ['EVENT_ca8e7a084029b8b7e02eb8842acb141f', 'EVENT_d2faf24fb4086f49cfef3e8014dd3781']
    -> ['EVENT_8e145a194b6c9259e961716524a7a4a9', 'EVENT_3c2d611ee315668a59e18c42c42d093c']
    -> ['EVENT_854e2675ef627a5052293dbc38d9a94c', 'EVENT_35de36460b07aa617fc730fb974fb329']
    -> ['EVENT_8ec16546e047fec7e22aa769ae267012', 'EVENT_7dcd4c35a18eafcd79bdd8f2ed8890d6']
  ... (showing 10 of 26)

### temporal_relations (6 entries)
  ** BEFORE: 208 pairs
    -> ['EVENT_101ac6c53adcbbeead2630740356836d', 'EVENT_3c2d611ee315668a59e18c42c42d093c']
    -> ['EVENT_8ec16546e047fec7e22aa769ae267012', 'EVENT_b0f5453c636e130e5185f688140a1cc4']
    -> ['EVENT_ca8e7a084029b8b7e02eb8842acb141f', 'EVENT_35de36460b07aa617fc730fb974fb329']
    -> ['TIME_df5124f8835b2fbcbe13ca4cf9bb8e82', 'EVENT_35de36460b07aa617fc730fb974fb329']
    -> ['EVENT_b0f5453c636e130e5185f688140a1cc4', 'EVENT_1793cae7aaf55ac974a98b49664311b9']
    -> ['EVENT_101ac6c53adcbbeead2630740356836d', 'EVENT_b374c762a61a1b07126c71525d63a4c5']
    -> ['EVENT_101ac6c53adcbbeead2630740356836d', 'EVENT_763ad23a774723f74a9ac6ca430a8d45']
    -> ['EVENT_f7dc4b4002a9bb1857f9c5943a7fa697', 'EVENT_3c2d611ee315668a59e18c42c42d093c']
    -> ['EVENT_d2faf24fb4086f49cfef3e8014dd3781', 'EVENT_101ac6c53adcbbeead2630740356836d']
    -> ['EVENT_f7dc4b4002a9bb1857f9c5943a7fa697', 'EVENT_763ad23a774723f74a9ac6ca430a8d45']
  ** OVERLAP: 0 pairs
  ** CONTAINS: 1 pairs
    -> ['TIME_df5124f8835b2fbcbe13ca4cf9bb8e82', 'EVENT_ca8e7a084029b8b7e02eb8842acb141f']
  ** SIMULTANEOUS: 1 pairs
    -> ['EVENT_35de36460b07aa617fc730fb974fb329', 'EVENT_763ad23a774723f74a9ac6ca430a8d45']
  ** ENDS-ON: 0 pairs
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 210)

---

## Sample 9: 1979 Kurdish rebellion in Iran
**Doc ID**: b403d8d5d5edc7b95460543b5c2af019

### Sentences (11 total)
  [0] The 1979 Kurdish rebellion in Iran erupted in mid-March 1979, some two months after the completion of the Iranian Revolution.
  [1] It subsequently became the largest among the nationwide uprisings in Iran against the new state and one of the most intense Kurdish rebellions in modern Iran.
  [2] Initially, Kurdish movements were trying to align with the new government of Iran, seeking to emphasize their Muslim identity and seek common ground with other Iranians.
  [3] KDPI even briefly branded itself as non-"separatist" organization, allegedly criticizing those calling for independence, but nevertheless calling for political autonomy.
  [4] However, relations between some Kurdish organizations and the Iranian government quickly deteriorated, and though Shi'a Kurds and some tribal leaders turned towards the new Shi'a Islamic State, Sunni 
  [5] While at first, Kurdish militants, primarily of the Democratic Party of Iranian Kurdistan, made some territorial gains in the area of Mahabad and ousted the Iranian troops from the region, a large sca
  [6] Following the eruption of the Iran–Iraq War in September 1980, an even greater effort was made by the Iranian government to crush the Kurdish rebellion, which was the only one of the 1979 uprisings to
  [7] By late 1980, the Iranian regular forces and the Revolutionary Guard ousted the Kurdish militants from their strongholds, but groups of Kurdish militants kept executing sporadic attacks against Irania
  [8] The clashes in the area went on as late as 1983.
  [9] About 10,000 people were killed in the course of the Kurdish rebellion, with 1,200 of them being Kurdish political prisoners, executed in the last phases of the rebellion, mostly by the Iranian govern
  [10] The Kurdish-Iranian dispute resurged only in 1989, following the assassination of a KDP-I leader.

### Events (32 total)
  - EVENT_9344355f7a13b807b35a036aff27f52a | type=Reforming_a_system | trigger='revolution' | sent_id=0 | offset=[20, 21]
  - EVENT_d27f82e8c5b1439d47cf7eadcf14b592 | type=Killing | trigger='executed' | sent_id=9 | offset=[22, 23]
  - EVENT_49562a2f1cfcae9596425c20d1f7b9b0 | type=Change_of_leadership | trigger='ousted' | sent_id=7 | offset=[12, 13]
  - EVENT_175376848084c69e73d6811c67135186 | type=Change_of_leadership | trigger='independence' | sent_id=3 | offset=[17, 18]
  - EVENT_1c96fdac601bce699b634a83362bbae0 | type=Killing | trigger='assassination' | sent_id=10 | offset=[10, 11]
  - EVENT_efb24a814a2c38ae153be485747ad3d1 | type=Cause_change_of_position_on_a_scale | trigger='deteriorated' | sent_id=4 | offset=[12, 13]
  - EVENT_0bf300d5f1a67673b1bf420cd341c883 | type=Killing | trigger='killed' | sent_id=9 | offset=[4, 5]
  - EVENT_a53b36fb5c5d88550d828bff2fdacf63 | type=Violence | trigger='uprisings' | sent_id=1 | offset=[8, 9]
  - EVENT_c5da7c6293a09ad15e1a4d889eb165db | type=Violence | trigger='clashes' | sent_id=8 | offset=[1, 2]
  - EVENT_1570bab48c09bb49514d90233b69085a | type=Judgment_communication | trigger='criticizing' | sent_id=3 | offset=[13, 14]
  - EVENT_da0a637e657d0bc3eb56c5e4cc336fb9 | type=Request | trigger='calling' | sent_id=3 | offset=[15, 16]
  - EVENT_b78f20576c6c9940fe14595530a6ef3a | type=Earnings_and_losses | trigger='gains' | sent_id=5 | offset=[19, 20]
  - EVENT_602d39f8922b685789aaca121f9c625b | type=Labeling | trigger='branded' | sent_id=3 | offset=[3, 4]
  - EVENT_d59bdda23f37a431ec336f83d3c697b3 | type=Hindering | trigger='subdued' | sent_id=6 | offset=[51, 52]
  - EVENT_df412cafae15f55c115ea1aee984d2ec | type=Change_of_leadership | trigger='ousted' | sent_id=5 | offset=[26, 27]
  - EVENT_46a1358bc8fe2b7eea6dbb1f1bea951c | type=Violence | trigger='conflict' | sent_id=5 | offset=[51, 52]
  - EVENT_ce7fbbb70f802284ee5ed8d1dcc61f71 | type=Recovering | trigger='resurged' | sent_id=10 | offset=[3, 4]
  - EVENT_35d985a3043b780c1a43d04c619dcf7d | type=Process_end | trigger='completion' | sent_id=0 | offset=[16, 17]
  - EVENT_82e508dbb75b14680e9be165ae4cfd35 | type=Violence | trigger='uprisings' | sent_id=6 | offset=[35, 36]
  - EVENT_b66837d8daaf5ef3413463692e982f74 | type=Cause_change_of_position_on_a_scale | trigger='reversed' | sent_id=5 | offset=[46, 47]
  - EVENT_2e9ea69130e8421c6282c32464874e6b | type=Violence | trigger='rebellions' | sent_id=6 | offset=[47, 48]
  - EVENT_8e9340a2cd61c42368acc1c919780f75 | type=Process_start | trigger='eruption' | sent_id=6 | offset=[2, 3]
  - EVENT_121c9f4ab29de9d746fcb253adb4b74f | type=Social_event | trigger='movements' | sent_id=2 | offset=[3, 4]
  - EVENT_af1120d6b981aa4c71f5af43217f1fdb | type=Becoming | trigger='became' | sent_id=1 | offset=[2, 3]
  - EVENT_26a818c3730bc14902fe4a91ed7f2cfa | type=Attack | trigger='attacks' | sent_id=7 | offset=[28, 29]
  - EVENT_e6970225fa4ba7ddafe4f0864c066ec6 | type=Violence | trigger='rebellion' | sent_id=0 | offset=[3, 4]
  - EVENT_e6970225fa4ba7ddafe4f0864c066ec6 | type=Violence | trigger='rebellion' | sent_id=6 | offset=[25, 26]
  - EVENT_e6970225fa4ba7ddafe4f0864c066ec6 | type=Violence | trigger='rebellion' | sent_id=9 | offset=[11, 12]
  - EVENT_e6970225fa4ba7ddafe4f0864c066ec6 | type=Violence | trigger='rebellion' | sent_id=9 | offset=[29, 30]
  - EVENT_ed20a0cd344e9d85ee878c90db3fb61f | type=Violence | trigger='rebellions' | sent_id=1 | offset=[22, 23]
  - EVENT_86ddd7a8f56cfd39fc74772560013168 | type=Request | trigger='calling' | sent_id=3 | offset=[21, 22]
  - EVENT_707a5f18da2228301fb6deb3439be2c7 | type=Cause_change_of_position_on_a_scale | trigger='emphasize' | sent_id=2 | offset=[17, 18]
  - EVENT_8bacf1dc79f6c2a09b88aeb9b80e55f0 | type=Becoming | trigger='turned' | sent_id=4 | offset=[24, 25]

### causal_relations (2 entries)
  ** CAUSE: 2 pairs
    -> ['EVENT_c5da7c6293a09ad15e1a4d889eb165db', 'EVENT_0bf300d5f1a67673b1bf420cd341c883']
    -> ['EVENT_ed20a0cd344e9d85ee878c90db3fb61f', 'EVENT_0bf300d5f1a67673b1bf420cd341c883']
  ** PRECONDITION: 7 pairs
    -> ['EVENT_c5da7c6293a09ad15e1a4d889eb165db', 'EVENT_d27f82e8c5b1439d47cf7eadcf14b592']
    -> ['EVENT_8e9340a2cd61c42368acc1c919780f75', 'EVENT_62c160309979333bd5665ca132e092df']
    -> ['EVENT_b78f20576c6c9940fe14595530a6ef3a', 'EVENT_df412cafae15f55c115ea1aee984d2ec']
    -> ['EVENT_ed20a0cd344e9d85ee878c90db3fb61f', 'EVENT_d27f82e8c5b1439d47cf7eadcf14b592']
    -> ['EVENT_ed20a0cd344e9d85ee878c90db3fb61f', 'EVENT_e6970225fa4ba7ddafe4f0864c066ec6']
    -> ['EVENT_121c9f4ab29de9d746fcb253adb4b74f', 'EVENT_707a5f18da2228301fb6deb3439be2c7']
    -> ['EVENT_a53b36fb5c5d88550d828bff2fdacf63', 'EVENT_d27f82e8c5b1439d47cf7eadcf14b592']

### temporal_relations (6 entries)
  ** BEFORE: 774 pairs
    -> ['TIME_0372df3baa32116bdd6e0cb20be55fde', 'EVENT_ce7fbbb70f802284ee5ed8d1dcc61f71']
    -> ['TIME_436cb24e34db6d22b32d81d4e10a70a8', 'EVENT_8bacf1dc79f6c2a09b88aeb9b80e55f0']
    -> ['EVENT_a53b36fb5c5d88550d828bff2fdacf63', 'EVENT_c5da7c6293a09ad15e1a4d889eb165db']
    -> ['EVENT_1570bab48c09bb49514d90233b69085a', 'EVENT_ce7fbbb70f802284ee5ed8d1dcc61f71']
    -> ['EVENT_86ddd7a8f56cfd39fc74772560013168', 'EVENT_d27f82e8c5b1439d47cf7eadcf14b592']
    -> ['EVENT_175376848084c69e73d6811c67135186', 'EVENT_b66837d8daaf5ef3413463692e982f74']
    -> ['EVENT_d59bdda23f37a431ec336f83d3c697b3', 'TIME_27c7735d3ec4d73d5effade1cfef3c45']
    -> ['EVENT_b66837d8daaf5ef3413463692e982f74', 'TIME_27c7735d3ec4d73d5effade1cfef3c45']
    -> ['EVENT_ed20a0cd344e9d85ee878c90db3fb61f', 'EVENT_86ddd7a8f56cfd39fc74772560013168']
    -> ['EVENT_a53b36fb5c5d88550d828bff2fdacf63', 'EVENT_0bf300d5f1a67673b1bf420cd341c883']
  ** OVERLAP: 0 pairs
  ** CONTAINS: 5 pairs
    -> ['TIME_2f954471ab59592935e61adfed08eaad', 'EVENT_8e9340a2cd61c42368acc1c919780f75']
    -> ['TIME_cb3d632b1680c5dc3922070cdf85de83', 'EVENT_ce7fbbb70f802284ee5ed8d1dcc61f71']
    -> ['TIME_0eb4793ec72cef0fa575a73a1aafc653', 'EVENT_df412cafae15f55c115ea1aee984d2ec']
    -> ['TIME_27c7735d3ec4d73d5effade1cfef3c45', 'EVENT_c5da7c6293a09ad15e1a4d889eb165db']
    -> ['TIME_0372df3baa32116bdd6e0cb20be55fde', 'EVENT_ea2401be9d27eeee5268602c511a19db']
  ** SIMULTANEOUS: 1 pairs
    -> ['EVENT_8bacf1dc79f6c2a09b88aeb9b80e55f0', 'EVENT_efb24a814a2c38ae153be485747ad3d1']
  ** ENDS-ON: 0 pairs
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 780)

---

## Sample 10: Death of Eric Garner
**Doc ID**: ed34034eebbb1fa5144e33bdc12b5936

### Sentences (22 total)
  [0] On July 17, 2014, Eric Garner died in the New York City borough of Staten Island after Daniel Pantaleo, a New York City Police Department (NYPD) officer, put him in a chokehold while arresting him.
  [1] Video footage of the incident generated widespread national attention and raised questions about the appropriate use of force by law enforcement.
  [2] NYPD officers approached Garner on July 17 on suspicion of selling single cigarettes from packs without tax stamps.
  [3] After Garner told the police that he was tired of being harassed and that he was not selling cigarettes, the officers attempted to arrest Garner.
  [4] When Officer Pantaleo placed his hands on Garner, Garner refused to cooperate and pulled his arms away.
  [5] Pantaleo then placed his arm around Garner's neck and wrestled him to the ground.
  [6] With multiple officers restraining him, Garner repeated the words "I can't breathe" 11 times while lying face down on the sidewalk.
  [7] After Garner lost consciousness, officers turned him onto his side to ease his breathing.
  [8] Garner remained lying on the sidewalk for seven minutes while the officers waited for an ambulance to arrive.
  [9] Garner was pronounced dead at an area hospital approximately one hour later.
  [10] Officer Pantaleo was placed on desk duty following Garner's death.
  [11] The medical examiner ruled Garner's death a homicide.
  [12] (According to the medical examiner's definition, a homicide is a death caused by the intentional actions of another person or persons; the use of the term does not necessarily mean that a crime was co
  [13] Specifically, an autopsy indicated that Garner's death resulted from "[compression] of neck (choke hold), compression of chest and prone positioning during physical restraint by police".
  [14] Asthma, heart disease, and obesity were cited as contributing factors.
  [15] On December 3, 2014, a Richmond County grand jury decided not to indict Officer Pantaleo.
  [16] This decision stirred public protests and rallies, with charges of police brutality made by protesters.
  [17] By December 28, 2014, at least 50 demonstrations had been held nationwide in response to the Garner case, while hundreds of demonstrations against general police brutality counted Garner as a focal po
  [18] On July 13, 2015, an out-of-court settlement was announced in which the City of New York would pay the Garner family $5.9 million.
  [19] In 2019, the U.S. Department of Justice declined to bring criminal charges against Pantaleo under federal civil rights laws.
  [20] A New York Police Department disciplinary hearing regarding Pantaleo's treatment of Garner was held in the summer of 2019; on August 2, 2019, an administrative judge recommended that Pantaleo's employ
  [21] Pantaleo was fired on August 19, 2019, five years after Garner's death.

### Events (61 total)
  - EVENT_d0c08dfe987fdadc9c1ea0308efd4c4e | type=Hindering | trigger='restraining' | sent_id=6 | offset=[3, 4]
  - EVENT_a6a2a148bfa5f74e5fbbb878c2a7c3f3 | type=Arrest | trigger='arrest' | sent_id=3 | offset=[24, 25]
  - EVENT_dbf65a0d628ca0833586651dd2c4280d | type=Creating | trigger='raised' | sent_id=1 | offset=[10, 11]
  - EVENT_73bcc28ecee50ada56601e5cf9dbdce0 | type=Bringing | trigger='bring' | sent_id=19 | offset=[10, 11]
  - EVENT_3d6fc557d16f71bb07f6ac7ac2c2b8fd | type=Deciding | trigger='decision' | sent_id=16 | offset=[1, 2]
  - EVENT_164d5d6c705ecec57b9e550aeba35d3e | type=Change_sentiment | trigger='harassed' | sent_id=3 | offset=[11, 12]
  - EVENT_4f89fdd22b253e15ce4254a58ae7c546 | type=Using | trigger='use' | sent_id=12 | offset=[26, 27]
  - EVENT_27eb64817cfab8053032fec68c0b7f45 | type=Statement | trigger='pronounced' | sent_id=9 | offset=[2, 3]
  - EVENT_5b22e2df951d9006deff6062e1fd16f6 | type=Motion | trigger='pulled' | sent_id=4 | offset=[14, 15]
  - EVENT_c45833f9a5904aac04d1f94fc94ac1af | type=Placing | trigger='positioning' | sent_id=13 | offset=[27, 28]
  - EVENT_7701f345ff777a05d86f6affd841d719 | type=Extradition | trigger='indict' | sent_id=15 | offset=[14, 15]
  - EVENT_9cb2b92bb7d3f6f55801eef9e635a68c | type=Process_end | trigger='fired' | sent_id=21 | offset=[2, 3]
  - EVENT_54f009601db11dccbcf5f5705245d64a | type=Motion | trigger='turned' | sent_id=7 | offset=[6, 7]
  - EVENT_f6ceba17ffdc89c8f1a26f75d154489e | type=Commitment | trigger='committed' | sent_id=12 | offset=[38, 39]
  - EVENT_cd3bc213e14cd6fb1d63d61c2f290ebd | type=Motion | trigger='stirred' | sent_id=16 | offset=[2, 3]
  - EVENT_968ef47f37c1b36dcf0275a982cd9cf3 | type=Defending | trigger='held' | sent_id=17 | offset=[12, 13]
  - EVENT_f66f9ff0260efe6fca78922a96cdb863 | type=Death | trigger='died' | sent_id=0 | offset=[8, 9]
  - EVENT_f52e6f66379ab88ff5c768c65daf021b | type=Arriving | trigger='arrive' | sent_id=8 | offset=[17, 18]
  - EVENT_68c92dac1a605dd38d6f393cdde22cf7 | type=Placing | trigger='lying' | sent_id=8 | offset=[2, 3]
  - EVENT_1d534ef4791aeb356b7c661f52d21b48 | type=Placing | trigger='placed' | sent_id=4 | offset=[3, 4]
  - EVENT_d1c4c9d507a75341c961a32837c8f0e8 | type=Agree_or_refuse_to_act | trigger='refused' | sent_id=4 | offset=[10, 11]
  - EVENT_88ad51abe76835c259e7da5d750ba1e3 | type=Defending | trigger='hold' | sent_id=13 | offset=[19, 20]
  - EVENT_8eace03bd2fac91b1d4bd2d53c740799 | type=Body_movement | trigger='wrestled' | sent_id=5 | offset=[10, 11]
  - EVENT_9314fda7ca20527573fbd992b0cf4eb4 | type=Commerce_sell | trigger='selling' | sent_id=3 | offset=[17, 18]
  - EVENT_0b3530475f015a3e524a745101c012a8 | type=Death | trigger='dead' | sent_id=9 | offset=[3, 4]
  - EVENT_98d0ef2df808c281240d4572f5fa00e4 | type=Causation | trigger='caused' | sent_id=12 | offset=[14, 15]
  - EVENT_12fcbf95f8bff037fd9e0628f34879ff | type=Placing | trigger='placed' | sent_id=5 | offset=[2, 3]
  - EVENT_b3ba1a6233afc53cbeff97ff12f7a088 | type=Expressing_publicly | trigger='announced' | sent_id=18 | offset=[10, 11]
  - EVENT_748ef33248d227f835dae58493bf76d4 | type=Legal_rulings | trigger='ruled' | sent_id=11 | offset=[3, 4]
  - EVENT_4fc7f580505c7f3dd8366c2451ae1c94 | type=Temporary_stay | trigger='remained' | sent_id=8 | offset=[1, 2]

### causal_relations (2 entries)
  ** CAUSE: 3 pairs
    -> ['EVENT_ee668ceb63f3b5fe0f533fd707c13fe5', 'EVENT_363844a80fd263c0d6ffeb18258b7a9a']
    -> ['EVENT_3d6fc557d16f71bb07f6ac7ac2c2b8fd', 'EVENT_cd3bc213e14cd6fb1d63d61c2f290ebd']
    -> ['EVENT_d0c08dfe987fdadc9c1ea0308efd4c4e', 'EVENT_87df9971f8bb157afc80b19d686be630']
  ** PRECONDITION: 40 pairs
    -> ['EVENT_78b3065f52b2a916cb2599e27516b449', 'EVENT_84892ec44addba79844d226bef863195']
    -> ['EVENT_d1c4c9d507a75341c961a32837c8f0e8', 'EVENT_5b22e2df951d9006deff6062e1fd16f6']
    -> ['EVENT_a6a2a148bfa5f74e5fbbb878c2a7c3f3', 'EVENT_d1c4c9d507a75341c961a32837c8f0e8']
    -> ['EVENT_54f009601db11dccbcf5f5705245d64a', 'EVENT_84892ec44addba79844d226bef863195']
    -> ['EVENT_78b3065f52b2a916cb2599e27516b449', 'EVENT_54f009601db11dccbcf5f5705245d64a']
    -> ['EVENT_1d534ef4791aeb356b7c661f52d21b48', 'EVENT_d0c08dfe987fdadc9c1ea0308efd4c4e']
    -> ['EVENT_a6a2a148bfa5f74e5fbbb878c2a7c3f3', 'EVENT_8eace03bd2fac91b1d4bd2d53c740799']
    -> ['EVENT_78b3065f52b2a916cb2599e27516b449', 'EVENT_b54899cfdba1d84b7b79080406735ec8']
    -> ['EVENT_cd3bc213e14cd6fb1d63d61c2f290ebd', 'EVENT_8634ee74a2f942bd9d694cda35fe2430']
    -> ['EVENT_4c5aa40d878f137b20401aa0e90f396b', 'EVENT_a6a2a148bfa5f74e5fbbb878c2a7c3f3']
  ... (showing 10 of 43)

### temporal_relations (6 entries)
  ** BEFORE: 1559 pairs
    -> ['EVENT_164d5d6c705ecec57b9e550aeba35d3e', 'EVENT_748ef33248d227f835dae58493bf76d4']
    -> ['EVENT_9314fda7ca20527573fbd992b0cf4eb4', 'EVENT_b54899cfdba1d84b7b79080406735ec8']
    -> ['EVENT_0b3530475f015a3e524a745101c012a8', 'EVENT_d53db99ae1bb52bcb38bf0eb29588b93']
    -> ['EVENT_b4336815f76f62a7caabeda95dbb408b', 'EVENT_27eb64817cfab8053032fec68c0b7f45']
    -> ['EVENT_7e3fcb81f45b8a133953127937577037', 'TIME_de2c09bcea014f293008dd57ed32ec13']
    -> ['TIME_791b6577a4b1906139d9aa30f5fd93b6', 'EVENT_62d339230c7e12fbdf1752d78fecf397']
    -> ['EVENT_34502b26e483231d2d450f25cb04df99', 'EVENT_7701f345ff777a05d86f6affd841d719']
    -> ['EVENT_ee668ceb63f3b5fe0f533fd707c13fe5', 'EVENT_82331e93b180ae9a3c4714a237004daf']
    -> ['TIME_b8cd236947a92b4f325bdd5ea5a0270a', 'EVENT_dbf65a0d628ca0833586651dd2c4280d']
    -> ['EVENT_a6a2a148bfa5f74e5fbbb878c2a7c3f3', 'EVENT_10336ac3b749a4c81f5dab56be865c76']
  ** OVERLAP: 3 pairs
    -> ['TIME_784b2c1810db99917849d7372cb74872', 'TIME_9f9ece1975c7137453505e4c67be74eb']
    -> ['EVENT_164d5d6c705ecec57b9e550aeba35d3e', 'EVENT_9314fda7ca20527573fbd992b0cf4eb4']
    -> ['TIME_ada63cd93faa4fd18d60fae088f4e373', 'TIME_9f9ece1975c7137453505e4c67be74eb']
  ** CONTAINS: 121 pairs
    -> ['TIME_9f9ece1975c7137453505e4c67be74eb', 'EVENT_d53db99ae1bb52bcb38bf0eb29588b93']
    -> ['EVENT_b3ba1a6233afc53cbeff97ff12f7a088', 'EVENT_82d764e2ceeda4b7b906f730bd07363b']
    -> ['TIME_ada63cd93faa4fd18d60fae088f4e373', 'EVENT_8eace03bd2fac91b1d4bd2d53c740799']
    -> ['TIME_784b2c1810db99917849d7372cb74872', 'EVENT_f52e6f66379ab88ff5c768c65daf021b']
    -> ['TIME_9f9ece1975c7137453505e4c67be74eb', 'EVENT_ee668ceb63f3b5fe0f533fd707c13fe5']
    -> ['TIME_dae69ce79be9e65064323d14320f2b01', 'EVENT_cd3bc213e14cd6fb1d63d61c2f290ebd']
    -> ['TIME_784b2c1810db99917849d7372cb74872', 'EVENT_54f009601db11dccbcf5f5705245d64a']
    -> ['TIME_784b2c1810db99917849d7372cb74872', 'EVENT_68c92dac1a605dd38d6f393cdde22cf7']
    -> ['TIME_ada63cd93faa4fd18d60fae088f4e373', 'EVENT_164d5d6c705ecec57b9e550aeba35d3e']
    -> ['TIME_ada63cd93faa4fd18d60fae088f4e373', 'EVENT_f66f9ff0260efe6fca78922a96cdb863']
  ** SIMULTANEOUS: 13 pairs
    -> ['EVENT_87df9971f8bb157afc80b19d686be630', 'EVENT_2ffc832a053cd0e8131959fdf5c0eca8']
    -> ['TIME_7631c75417a1df409d1033e97cf04512', 'EVENT_4fc7f580505c7f3dd8366c2451ae1c94']
    -> ['EVENT_247e6e063d233d7cdbf28ed728c5bc19', 'EVENT_84892ec44addba79844d226bef863195']
    -> ['EVENT_8634ee74a2f942bd9d694cda35fe2430', 'EVENT_7e3fcb81f45b8a133953127937577037']
    -> ['EVENT_1dd4cc50ae3bf89bdd11c9885da1b5cc', 'EVENT_10336ac3b749a4c81f5dab56be865c76']
    -> ['EVENT_d1c4c9d507a75341c961a32837c8f0e8', 'EVENT_413bd1b416acfe51cc79aa5f090879d8']
    -> ['EVENT_73bcc28ecee50ada56601e5cf9dbdce0', 'EVENT_0c90ff74898692e1d82f05074c88eec5']
    -> ['EVENT_68c92dac1a605dd38d6f393cdde22cf7', 'EVENT_b54899cfdba1d84b7b79080406735ec8']
    -> ['EVENT_68c92dac1a605dd38d6f393cdde22cf7', 'EVENT_4fc7f580505c7f3dd8366c2451ae1c94']
    -> ['EVENT_b54899cfdba1d84b7b79080406735ec8', 'EVENT_4fc7f580505c7f3dd8366c2451ae1c94']
  ** ENDS-ON: 3 pairs
    -> ['EVENT_27eb64817cfab8053032fec68c0b7f45', 'TIME_9f9ece1975c7137453505e4c67be74eb']
    -> ['TIME_791b6577a4b1906139d9aa30f5fd93b6', 'TIME_dae69ce79be9e65064323d14320f2b01']
    -> ['EVENT_0b3530475f015a3e524a745101c012a8', 'TIME_9f9ece1975c7137453505e4c67be74eb']
  ** BEGINS-ON: 0 pairs
  ... (showing 10 of 1699)

### subevent_relations (4 entries)
  -> ['EVENT_34502b26e483231d2d450f25cb04df99', 'EVENT_8eace03bd2fac91b1d4bd2d53c740799']
  -> ['EVENT_34502b26e483231d2d450f25cb04df99', 'EVENT_1d534ef4791aeb356b7c661f52d21b48']
  -> ['EVENT_34502b26e483231d2d450f25cb04df99', 'EVENT_d1c4c9d507a75341c961a32837c8f0e8']
  -> ['EVENT_34502b26e483231d2d450f25cb04df99', 'EVENT_d0c08dfe987fdadc9c1ea0308efd4c4e']

---