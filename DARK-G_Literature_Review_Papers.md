

## THEME 1: Cybersecurity Knowledge Graphs and Ontologies

### Foundational Surveys

**1. "A survey on cybersecurity knowledge graph construction"**
- **Authors:** Multiple authors
- **Source:** Computers & Security (ScienceDirect), October 2023
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0167404823004340
- **Relevance:** Comprehensive survey covering ontology-based creation, semantic triple extraction, and CKG construction methodologies. Directly applicable to DARK-G's knowledge acquisition pipeline.

**2. "Knowledge Graphs and Semantic Web Tools in Cyber Threat Intelligence: A Systematic Literature Review"**
- **Authors:** Anastasiadis, Bratsas, Ougiaroglou et al.
- **Source:** J. Cybersecurity and Privacy, August 2024
- **URL:** https://www.mdpi.com/2624-800X/4/3/25
- **Relevance:** Systematic review of 225+ papers on ontologies and KGs in CTI domain. Covers interoperability, SOC integration, ML/DL augmentation. Essential for positioning DARK-G.

**3. "Cybersecurity knowledge graphs"**
- **Authors:** Multiple authors
- **Source:** Knowledge and Information Systems (Springer), April 2023
- **URL:** https://link.springer.com/article/10.1007/s10115-023-01860-3
- **Relevance:** Reviews graph-based data models, knowledge organization systems, and how KGs enable ML and automated reasoning over cyber-knowledge. Covers TAGraph, D3FEND.

**4. "Cybersecurity knowledge graphs construction and quality assessment"**
- **Authors:** Multiple authors
- **Source:** Complex & Intelligent Systems (Springer), August 2023
- **URL:** https://link.springer.com/article/10.1007/s40747-023-01205-1
- **Relevance:** Presents CS13K dataset (first manually constructed CKG dataset), expanded UCO ontology (8→16 categories). Quality assessment methods applicable to DARK-G.

**5. "An Overview of Cybersecurity Knowledge Graphs"**
- **Authors:** UMBC Ebiquity group
- **Source:** UMBC Technical Report
- **URL:** https://ebiquity.umbc.edu/get/a/publication/1210.pdf
- **Relevance:** Reviews 40+ manuscripts on CKGs, discusses UCO/UCO 2.0, ATT&CK mapping, ICS defense applications.

### Specific Frameworks and Tools

**6. "Toward a Knowledge Graph of Cybersecurity Countermeasures" (D3FEND)**
- **Authors:** Kaloroumakis et al. (MITRE)
- **Source:** MITRE Technical Paper
- **URL:** https://d3fend.mitre.org/resources/D3FEND.pdf
- **Relevance:** Describes D3FEND knowledge graph of defensive techniques. Semantic technique representation, Digital Artifact Ontology. Key alignment target for DARK-G defense ontology.

**7. "Actionable Cyber Threat Intelligence using Knowledge Graphs and Large Language Models"**
- **Authors:** Multiple authors
- **Source:** WACCO 2024 Workshop (arXiv:2407.02528)
- **URL:** https://arxiv.org/abs/2407.02528
- **Relevance:** Explores LLM-based triple extraction for CTI, guidance framework, fine-tuning for ontology-adherent extraction. Directly applicable to DARK-G's LLM extraction pipeline.

---

## THEME 2: LLM-Based Knowledge Graph Construction

### Comprehensive Surveys

**8. "LLM-empowered knowledge graph construction: A survey"**
- **Authors:** Multiple authors
- **Source:** arXiv:2510.20345, October 2025
- **URL:** https://arxiv.org/abs/2510.20345
- **Relevance:** Comprehensive survey on LLM-driven KG construction. Covers ontology engineering, knowledge extraction, knowledge fusion. Schema-based vs schema-free paradigms. Essential methodological foundation.

**9. "LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities"**
- **Authors:** Zhu, Zhang et al.
- **Source:** arXiv:2305.13168, December 2024
- **URL:** https://arxiv.org/abs/2305.13168
- **Relevance:** Evaluates GPT-4 for KG construction/reasoning across 8 datasets. Proposes AutoKG (multi-agent approach). Finds LLMs better as inference assistants than few-shot extractors.

**10. "Knowledge Graph Construction: Extraction, Learning, and Evaluation"**
- **Authors:** Multiple authors
- **Source:** Applied Sciences (MDPI), March 2025
- **URL:** https://www.mdpi.com/2076-3417/15/7/3727
- **Relevance:** Overview of 2022-2024 KG construction research. Covers extraction methods, learning paradigms, evaluation methodology. Addresses LLM hallucination mitigation.

**11. "Large Language Models for Knowledge Graph Embedding: A Survey"**
- **Authors:** Multiple authors
- **Source:** Mathematics (MDPI), July 2025
- **URL:** https://www.mdpi.com/2227-7390/13/14/2244
- **Relevance:** Survey on LLM methods for KG embedding. Covers prompt-based methods, pre-training, fine-tuning approaches. Applicable to DARK-G's entity/relation embedding.

**12. "Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey"**
- **Authors:** ZJU KG Group
- **Source:** GitHub repository with IJCAI/AAAI/ICLR papers
- **URL:** https://github.com/zjukg/KG-MM-Survey
- **Relevance:** Covers multimodal KG construction, entity extraction, relation extraction. Relevant for extending DARK-G to multimodal threat data.

---

## THEME 3: Adversarial Machine Learning Surveys

### Comprehensive Attack/Defense Surveys

**13. "Defense strategies for Adversarial Machine Learning: A survey"**
- **Authors:** Multiple authors
- **Source:** Computer Science Review (ScienceDirect), August 2023
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S1574013723000400
- **Relevance:** First domain-agnostic taxonomy of AML defenses. Covers cybersecurity, computer vision, NLP, audio domains. Defense categorization directly applicable to DARK-G ontology.

**14. "A System-Driven Taxonomy of Attacks and Defenses in Adversarial Machine Learning"**
- **Authors:** Sadeghi et al.
- **Source:** IEEE TETCI / PMC, 2020
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7971418/
- **Relevance:** Fine-grained taxonomy for ML system specification. Proposes adversarial ML cycle model. Arms race generation number metric. Foundational for DARK-G attack/defense modeling.

**15. "Adversarial examples: A survey of attacks and defenses in deep learning-enabled cybersecurity systems"**
- **Authors:** Multiple authors
- **Source:** Expert Systems with Applications (ScienceDirect), October 2023
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0957417423027252
- **Relevance:** Establishes taxonomy of DL cybersecurity applications. Reviews adversarial attacks/defenses for malware detection, NIDS, CPS, fraud detection. Curated dataset list.

**16. "A meta-survey of adversarial attacks against artificial intelligence algorithms"**
- **Authors:** Multiple authors
- **Source:** Neurocomputing (ScienceDirect), August 2025
- **URL:** https://www.sciencedirect.com/science/article/pii/S0925231225019034
- **Relevance:** Umbrella review synthesizing multiple systematic reviews. Covers gradient-based, transfer-based, score-based, decision-based, poisoning, privacy attacks. PICO framework analysis.

**17. "Defenses in Adversarial Machine Learning: A Survey"**
- **Authors:** Wu et al.
- **Source:** arXiv:2312.08890, December 2023
- **URL:** https://arxiv.org/abs/2312.08890
- **Relevance:** Life-cycle perspective on defenses (pre-training, training, post-training, deployment, inference). Covers backdoor, weight, adversarial example defenses. Unified taxonomy.

**18. "Adversarial attacks and defenses in explainable artificial intelligence: A survey"**
- **Authors:** Baniecki et al.
- **Source:** Information Fusion (ScienceDirect), February 2024
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S1566253524000812
- **Relevance:** Covers adversarial attacks on XAI/explanations. Unified notation and taxonomy. Relevant for trustworthy AI aspects of DARK-G.

**19. "A survey on adversarial attacks in computer vision: Taxonomy, visualization and future directions"**
- **Authors:** Long et al.
- **Source:** Computers & Security (ScienceDirect), July 2022
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0167404822002413
- **Relevance:** Taxonomy-based review of CV attack strategies. Robustness evaluation approaches. Defense from architectural perspective.

**20. "A Survey of Adversarial Examples in Computer Vision: Attack, Defense, and Beyond"**
- **Authors:** Multiple authors
- **Source:** Wuhan University Journal of Natural Sciences, March 2025
- **URL:** https://wujns.edpsciences.org/articles/wujns/full_html/2025/01/wujns-1007-1202-2025-01-0001-20/wujns-1007-1202-2025-01-0001-20.html
- **Relevance:** Latest comprehensive survey. Covers theoretical explanations, accuracy-robustness tradeoff, benign adversarial attacks.

---

## THEME 4: MITRE ATLAS and Threat Frameworks

**21. MITRE ATLAS Official Documentation**
- **Source:** MITRE Corporation
- **URL:** https://atlas.mitre.org/
- **Relevance:** Official adversarial ML threat matrix. 15 tactics, 66 techniques, 46 sub-techniques, 26 mitigations, 33 case studies. Primary alignment target for DARK-G ontology.

**22. "MITRE Adversarial Threat Landscape for AI Systems (ATLAS) Fact Sheet"**
- **Source:** MITRE Corporation
- **URL:** https://atlas.mitre.org/pdf-files/MITRE_ATLAS_Fact_Sheet.pdf
- **Relevance:** Official overview of ATLAS framework, TTPs, collaboration model.

**23. "ATLAS Overview" (NIST Presentation)**
- **Authors:** Dr. Christina Liaghati (MITRE ATLAS Lead)
- **Source:** NIST CSRC, September 2025
- **URL:** https://csrc.nist.gov/csrc/media/Presentations/2025/mitre-atlas/TuePM2.1-MITRE%20ATLAS%20Overview%20Sept%202025.pdf
- **Relevance:** Latest ATLAS updates including Morris II worm case study, RAG vulnerabilities.

**24. "Adversarial Threat Landscape for AI Systems" (GitHub)**
- **Source:** MITRE/Microsoft collaboration
- **URL:** https://github.com/mitre/advmlthreatmatrix
- **Relevance:** Original adversarial ML threat matrix repository. ATT&CK-style framework for ML threats.

---

## THEME 5: Physical Adversarial Attacks

### Adversarial Camouflage and Patches

**25. "From 2D-Patch to 3D-Camouflage: A Review of Physical Adversarial Attack in Object Detection"**
- **Authors:** Multiple authors
- **Source:** Electronics (MDPI), October 2025
- **URL:** https://www.mdpi.com/2079-9292/14/21/4236
- **Relevance:** Systematic categorization of physical attacks: 2D manipulation, signal injection, 3D camouflage. Nine key attributes analysis. Essential for defence camouflage case study.

**26. "Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies"**
- **Authors:** Multiple authors
- **Source:** arXiv:2202.08892, May 2022
- **URL:** https://arxiv.org/abs/2202.08892
- **Relevance:** **Directly relevant to defence case study.** Imperceptible patches for military assets (aircraft). ISR technology evasion. Colour perceptibility constraints.

**27. "TACO: Adversarial Camouflage Optimization on Trucks to Fool Object Detectors"**
- **Authors:** Multiple authors
- **Source:** arXiv:2410.21443, October 2024
- **URL:** https://arxiv.org/html/2410.21443v1
- **Relevance:** Unreal Engine 5 integration, differentiable rendering, YOLOv8 targeting. Transferability to DETR, Faster R-CNN. Methodology applicable to DARK-G simulation.

**28. "Naturalistic physical adversarial camouflage for object detection via differentiable rendering and style learning"**
- **Authors:** Multiple authors
- **Source:** Pattern Recognition (ScienceDirect), October 2025
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0031320325012841
- **Relevance:** GAN-Based Style Learner (GBSL), optical imaging physical model. Addresses perceptual plausibility and physical robustness.

**29. "Rust-Style Patch: A Physical and Naturalistic Camouflage Attacks on Object Detector for Remote Sensing Images"**
- **Authors:** Multiple authors
- **Source:** Remote Sensing (MDPI), February 2023
- **URL:** https://www.mdpi.com/2072-4292/15/4/885
- **Relevance:** Remote sensing specific. Military asset concealment. Style-transfer based natural patches. Directly applicable to defence case study.

**30. "Learning Coated Adversarial Camouflages for Object Detectors"**
- **Authors:** Duan et al.
- **Source:** IJCAI 2022
- **URL:** https://www.ijcai.org/proceedings/2022/0125.pdf
- **Relevance:** CAC framework for arbitrary objects. Unity simulation engine. 3D-printed adversarial vehicle validation.

**31. "Physical Adversarial Attack on Vehicle Detectors" (ICLR)**
- **Source:** OpenReview/ICLR
- **URL:** https://openreview.net/pdf?id=SJgEl3A5tm
- **Relevance:** Adversarial camouflage for vehicle detection evasion. Unreal simulation. Photo-realistic rendering. EoT principle.

**32. "Deep learning adversarial attacks and defenses in autonomous vehicles: A systematic literature review from a safety perspective"**
- **Authors:** Multiple authors
- **Source:** Artificial Intelligence Review (Springer), November 2024
- **URL:** https://link.springer.com/article/10.1007/s10462-024-11014-8
- **Relevance:** SOTIF-inspired taxonomy. Camera, LiDAR, sensor fusion attacks. Camouflage scenarios. Comprehensive AV safety perspective.

**33. "AdvSign: Artistic advertising sign camouflage for physical attacking object detector"**
- **Authors:** Multiple authors
- **Source:** Neural Networks (ScienceDirect), February 2025
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0893608025001509
- **Relevance:** Artistic camouflage patterns. CARLA simulator integration. Mobile carrier attacks (vehicles, bags).

---

## THEME 6: Adversarial Attacks on Intrusion Detection Systems

**34. "Adversarial Machine Learning Attacks against Intrusion Detection Systems: A Survey on Strategies and Defense"**
- **Authors:** Rassam et al.
- **Source:** Future Internet (MDPI), January 2023
- **URL:** https://www.mdpi.com/1999-5903/15/2/62
- **Relevance:** Comprehensive IDS-focused AML survey. White-box/black-box attacks. GAN-based evasion. Defense strategies. Benchmark datasets.

**35. "Adversarial attacks on machine learning cybersecurity defences in Industrial Control Systems"**
- **Authors:** Multiple authors
- **Source:** Journal of Information Security and Applications (ScienceDirect), February 2021
- **URL:** https://www.sciencedirect.com/science/article/pii/S2214212620308607
- **Relevance:** **Critical for ICS relevance.** JSMA attacks against ICS-specific ML-IDS. Severe consequence analysis. Directly applicable to DARK-G ICS focus.

**36. "A Systematic Study of Adversarial Attacks Against Network Intrusion Detection Systems"**
- **Authors:** Multiple authors
- **Source:** Electronics (MDPI), December 2024
- **URL:** https://www.mdpi.com/2079-9292/13/24/5030
- **Relevance:** Systematic examination of image-domain attacks against ML-NIDS. PGD, ZOO, Boundary, HopSkipJump attacks. NSL-KDD evaluation.

**37. "Adversarial attacks against supervised machine learning based network intrusion detection systems"**
- **Authors:** Multiple authors
- **Source:** PLOS ONE, October 2022
- **URL:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0275971
- **Relevance:** GAN-based poisoning and evasion attacks. CICIDS2017 evaluation. Decision Tree and Logistic Regression targeting.

**38. "Model Evasion Attack on Intrusion Detection Systems using Adversarial Machine Learning"**
- **Authors:** Ayub et al.
- **Source:** IEEE CISS 2020
- **URL:** https://ieeexplore.ieee.org/document/9086268/
- **Relevance:** JSMA against MLP-based IDS. CICIDS2017 and TRAbID2017 datasets. 22-30% precision drop demonstration.

**39. "An enhanced ensemble defense framework for boosting adversarial robustness of intrusion detection systems"**
- **Authors:** Multiple authors
- **Source:** Scientific Reports (Nature), April 2025
- **URL:** https://www.nature.com/articles/s41598-025-94023-z
- **Relevance:** Latest defense framework. Adversarial training, label smoothing, Gaussian augmentation. FGSM, BIM, JSMA, DeepFool evaluation.

**40. "Modeling Realistic Adversarial Attacks against Network Intrusion Detection Systems"**
- **Authors:** Multiple authors
- **Source:** Digital Threats: Research and Practice (ACM), 2021
- **URL:** https://dl.acm.org/doi/10.1145/3469659
- **Relevance:** Models realistic attacker capabilities. Feasibility analysis of adversarial attacks. Important for threat modeling realism.

**41. "Adversarial Attacks Against Deep Learning-Based Network Intrusion Detection Systems and Defense Mechanisms" (TIKI-TAKA)**
- **Source:** IEEE/ACM Transactions on Networking
- **URL:** https://ieeexplore.ieee.org/document/9674195/
- **Relevance:** TIKI-TAKA framework for robustness assessment. Five attack types against three NN detectors. Defense mechanisms.

---

## THEME 7: Semantic Web and OWL Reasoning

**42. "Introduction to the Semantic Web" (GraphDB Documentation)**
- **Source:** Ontotext GraphDB
- **URL:** https://graphdb.ontotext.com/documentation/10.8/introduction-to-semantic-web.html
- **Relevance:** Comprehensive introduction to RDF, RDFS, OWL, SPARQL. Forward/backward chaining inference. Technical foundation for DARK-G reasoning.

**43. "Reasoning with Big Knowledge Graphs: Choices, Pitfalls and Proven Recipes"**
- **Authors:** Ontotext
- **Source:** Ontotext Blog, November 2024
- **URL:** https://www.ontotext.com/blog/reasoning-with-big-knowledge-graphs/
- **Relevance:** Practical OWL reasoning at scale. Inference optimization. British Museum case study. 2.2B explicit + 328M inferred statements example.

**44. "Knowledge Graphs: Semantic Reasoning Meets Graph Architecture"**
- **Source:** RushDB, July 2025
- **URL:** https://rushdb.com/blog/knowledge-graphs-semantic-reasoning-meets-graph-architecture
- **Relevance:** Comparison of LPG vs semantic KG. Automatic classification via reasoning. Bio2RDF integration example.

**45. "GO, RDF/OWL and SPARQL" (Gene Ontology)**
- **Source:** Gene Ontology Consortium
- **URL:** https://geneontology.org/docs/sparql
- **Relevance:** Practical example of OWL reasoning for knowledge derivation. SPARQL query patterns. Quad store implementation.

---

## THEME 8: Australian Context and Defence Applications

**46. MITRE Secure AI Program**
- **Note:** Collaboration of 16 organizations including Microsoft, CrowdStrike, JPMorgan Chase
- **Relevance:** Industry collaboration model applicable to DARK-G partnership structure.

**47. Australian Cyber Security Strategy 2023-2030**
- **Source:** Australian Government
- **Relevance:** Six cyber shields framework. Shield 4 (Protected Critical Infrastructure) and Shield 5 (Sovereign Capabilities) alignment.

**48. SOCI Act 2018 (Security of Critical Infrastructure)**
- **Source:** Australian Government
- **Relevance:** Mandatory CIRMP requirements for critical infrastructure. AI/ML system risk assessment requirements.

**49. Cyber Security Act 2024**
- **Source:** Australian Government
- **Relevance:** Mandatory incident reporting. Proactive assessment requirements.

**50. AUKUS Pillar II - RAAIT (Resilient and Autonomous AI Technologies)**
- **Source:** Defence.gov.au
- **Relevance:** Adversarial robustness focus. Autonomous systems protection. Direct alignment for DARK-G defence applications.

---

## Summary: Key Papers by Application Area

### For Ontology Design Section:
- Papers 1-7 (CKG surveys and D3FEND)
- Papers 42-45 (Semantic Web foundations)

### For LLM Extraction Pipeline:
- Papers 7-12 (LLM-KG construction)

### For Attack Taxonomy:
- Papers 13-20 (AML surveys)
- Papers 21-24 (MITRE ATLAS)

### For Defence Case Study:
- Papers 25-33 (Physical adversarial attacks)
- Paper 26 especially (Military asset camouflage)

### For ICS/IDS Application:
- Papers 34-41 (IDS adversarial attacks)
- Paper 35 especially (ICS-specific)

### For Australian Context:
- Papers 46-50 (Policy and strategic alignment)

