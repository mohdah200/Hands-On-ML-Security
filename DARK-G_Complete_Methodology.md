# DARK-G: Complete Methodology and Solution Pipeline

## Executive Summary

**The Problem**: Every day, new adversarial attacks against ML models are published. Organisations cannot keep up - they don't know which attacks affect their models, which defenses work, or how to prioritise protection.

**The Solution**: DARK-G uses a **Semantic Knowledge Graph** to automatically:
1. Collect and structure adversarial ML knowledge from publications
2. Understand relationships between attacks, defenses, and model architectures
3. Reason about which threats apply to specific deployed models
4. Recommend defenses based on evidence
5. Alert when new threats emerge

**Why Semantic Knowledge Graphs?** Unlike databases that just store data, semantic KGs can **reason** and **infer** new knowledge. This is the critical difference that enables automated threat assessment.

---

# SECTION 1: THE PROBLEM IN DETAIL

## 1.1 The Adversarial ML Knowledge Explosion

```
Publication Rate of Adversarial ML Papers:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2015: ████ (~50 papers)
2018: ████████████ (~300 papers)
2020: ████████████████████ (~800 papers)
2023: ████████████████████████████████ (~1,500 papers)
2025: ████████████████████████████████████████ (~2,000+ papers)
```

**No human can read 2,000 papers per year and understand which attacks matter for their specific systems.**

## 1.2 Why Current Solutions Fail

### Approach 1: Manual Literature Review
```
Process:
  Security analyst reads papers → Takes notes → Creates report

Problems:
  ✗ Takes weeks/months
  ✗ Immediately outdated
  ✗ Different analysts reach different conclusions
  ✗ Cannot scale to thousands of papers
  ✗ Knowledge stays in analyst's head or static documents
```

### Approach 2: Traditional Databases
```
Process:
  Store attacks in database table → Query by name or type

Example Database:
┌─────────────┬──────────┬──────────┬─────────────┐
│ attack_name │ type     │ year     │ target      │
├─────────────┼──────────┼──────────┼─────────────┤
│ FGSM        │ evasion  │ 2015     │ classifiers │
│ PGD         │ evasion  │ 2018     │ classifiers │
│ AutoAttack  │ evasion  │ 2020     │ classifiers │
└─────────────┴──────────┴──────────┴─────────────┘

Problems:
  ✗ No relationships between attacks
  ✗ Cannot answer: "Is my ResNet50 vulnerable to AutoAttack?"
  ✗ Cannot infer: "If FGSM affects CNNs, does it affect my CNN?"
  ✗ Cannot reason about defense effectiveness
  ✗ Just stores facts - cannot derive new knowledge
```

### Approach 3: LLM Assistants (ChatGPT, etc.)
```
Process:
  Ask LLM "What attacks affect my model?" → Get response

Problems:
  ✗ Hallucinations - makes up attacks that don't exist
  ✗ No provenance - cannot trace where information came from
  ✗ Outdated knowledge - training cutoff misses recent attacks
  ✗ No systematic coverage - may miss critical threats
  ✗ Cannot integrate with your specific deployment context
```

## 1.3 What We Actually Need

```
REQUIREMENTS FOR EFFECTIVE ADVERSARIAL ML THREAT MANAGEMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONTINUOUS INGESTION
   Automatically collect new attacks as they are published
   
2. STRUCTURED REPRESENTATION  
   Store attacks, defenses, models in a way that captures relationships
   
3. REASONING CAPABILITY
   Automatically determine: "Attack X affects Architecture Y, 
   my model uses Architecture Y, therefore my model may be vulnerable"
   
4. CONTEXTUAL QUERIES
   Answer: "What threatens MY specific model in MY deployment context?"
   
5. DEFENSE MAPPING
   Know which defenses mitigate which attacks, with effectiveness ratings
   
6. PROVENANCE TRACKING
   Trace every fact back to its source (paper, CVE, incident report)
   
7. CONTINUOUS MONITORING
   Alert when new threats emerge that affect deployed systems
```

**A Semantic Knowledge Graph provides ALL of these capabilities.**

---

# SECTION 2: HOW SEMANTIC KNOWLEDGE GRAPHS WORK

## 2.1 The Core Concept: Knowledge as a Graph

Instead of storing data in tables, we store **knowledge as a network of connected facts**.

```
TRADITIONAL DATABASE (Disconnected Facts):
┌─────────────────────────────────────────────────────────┐
│ Table: Attacks          Table: Models                   │
│ ┌───────┬────────┐      ┌─────────┬────────────┐       │
│ │ FGSM  │ evasion│      │ MyModel │ ResNet50   │       │
│ │ PGD   │ evasion│      │ Model2  │ VGG16      │       │
│ └───────┴────────┘      └─────────┴────────────┘       │
│                                                         │
│ Problem: How do we know FGSM affects MyModel?          │
│ Answer: We don't - no relationship is stored.          │
└─────────────────────────────────────────────────────────┘

KNOWLEDGE GRAPH (Connected Knowledge):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    [FGSM] ───targets───> [CNN Architecture]            │
│      │                         ↑                        │
│      │                         │                        │
│   variant_of              is_a_type_of                 │
│      │                         │                        │
│      ↓                         │                        │
│    [PGD] ───targets───> [ResNet50] <───uses─── [MyModel]│
│                                                         │
│ Now we can TRAVERSE the graph:                         │
│ MyModel → uses → ResNet50 → is_type_of → CNN ← targets ← FGSM │
│ Therefore: FGSM may affect MyModel (INFERRED!)         │
└─────────────────────────────────────────────────────────┘
```

## 2.2 What Makes It "Semantic"?

"Semantic" means the graph understands **meaning**, not just data.

### Layer 1: The Ontology (Schema + Rules)

The **ontology** defines:
- What types of things exist (classes)
- What properties they can have
- What relationships are valid
- What rules govern inference

```
EXAMPLE ONTOLOGY STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASS HIERARCHY (What types of things exist):

                    [Attack]
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
  [Evasion]      [Poisoning]     [Extraction]
       │
  ┌────┴────┐
  ↓         ↓
[Gradient] [Physical]
  │           │
  ↓           ↓
[FGSM]    [AdvPatch]
[PGD]     [DAAC]

PROPERTY DEFINITIONS (What attributes things can have):

  Attack:
    - has perturbation_type (L2, L-infinity, L0)
    - has success_rate (0.0 to 1.0)
    - has computational_cost (low, medium, high)
    - published_year (integer)
    
  Defense:
    - has effectiveness (0.0 to 1.0)
    - has overhead (computational cost)
    - has accuracy_tradeoff (accuracy drop when applied)

RELATIONSHIP DEFINITIONS (How things can connect):

  Attack ──targets──> Architecture
  Defense ──mitigates──> Attack
  Attack ──bypasses──> Defense
  Model ──uses──> Architecture
  Model ──deployed_in──> Sector
  Attack ──variant_of──> Attack
```

### Layer 2: The Facts (Instance Data)

Actual knowledge stored as **triples** (Subject → Relationship → Object):

```
EXAMPLE FACTS (Triples):
━━━━━━━━━━━━━━━━━━━━━━━━

Subject              Relationship        Object
─────────────────────────────────────────────────
FGSM                 type                GradientAttack
FGSM                 targets             CNN
FGSM                 perturbation_type   L-infinity
FGSM                 published_year      2015

PGD                  type                GradientAttack
PGD                  variant_of          FGSM
PGD                  targets             CNN

AdversarialTraining  type                ProactiveDefense
AdversarialTraining  mitigates           FGSM
AdversarialTraining  mitigates           PGD
AdversarialTraining  effectiveness_vs    FGSM: 0.7

ResNet50             type                CNN
YOLOv5               type                CNN

MyDroneDetector      uses                YOLOv5
MyDroneDetector      deployed_in         Defense_Sector
MyDroneDetector      has_defense         InputNormalization
```

### Layer 3: The Reasoning Rules (Inference)

**This is the key innovation.** Rules that derive NEW knowledge automatically:

```
INFERENCE RULES:
━━━━━━━━━━━━━━━━

RULE 1: Architecture Vulnerability Propagation
─────────────────────────────────────────────────
IF:   Attack A targets Architecture X
AND:  Model M uses Architecture Y
AND:  Y is a subtype of X
THEN: Model M is potentially vulnerable to Attack A

Example Application:
  Fact: FGSM targets CNN
  Fact: YOLOv5 is a type of CNN
  Fact: MyDroneDetector uses YOLOv5
  ───────────────────────────────
  INFERRED: MyDroneDetector is potentially vulnerable to FGSM
  
  (We never explicitly stored this - the system figured it out!)


RULE 2: Variant Vulnerability Inheritance
─────────────────────────────────────────────────
IF:   Attack A2 is variant_of Attack A1
AND:  Attack A1 targets Architecture X
THEN: Attack A2 also targets Architecture X

Example Application:
  Fact: PGD is variant_of FGSM
  Fact: FGSM targets CNN
  ───────────────────────────────
  INFERRED: PGD targets CNN


RULE 3: Defense Gap Detection
─────────────────────────────────────────────────
IF:   Model M is potentially vulnerable to Attack A
AND:  Model M does NOT have a Defense D where D mitigates A
THEN: Model M has a defense gap for Attack A

Example Application:
  Inferred: MyDroneDetector vulnerable to AdversarialPatch
  Fact: MyDroneDetector has_defense InputNormalization
  Fact: InputNormalization does NOT mitigate AdversarialPatch
  ───────────────────────────────
  INFERRED: MyDroneDetector has defense gap for AdversarialPatch
  → GENERATE ALERT


RULE 4: Defense Recommendation
─────────────────────────────────────────────────
IF:   Model M has defense gap for Attack A
AND:  Defense D mitigates Attack A
AND:  Defense D has effectiveness > 0.5
THEN: Recommend Defense D for Model M

Example Application:
  Inferred: MyDroneDetector has defense gap for AdversarialPatch
  Fact: DetectorEnsemble mitigates AdversarialPatch
  Fact: DetectorEnsemble effectiveness_vs AdversarialPatch: 0.8
  ───────────────────────────────
  RECOMMENDATION: Deploy DetectorEnsemble for MyDroneDetector
```

## 2.3 The Complete Picture: How Inference Works

```
INFERENCE ENGINE PROCESS:
━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Load Explicit Facts
┌─────────────────────────────────────────────────────────┐
│ • FGSM targets CNN                                      │
│ • PGD variant_of FGSM                                   │
│ • YOLOv5 is_type_of CNN                                │
│ • MyDroneDetector uses YOLOv5                          │
│ • MyDroneDetector deployed_in Defense_Sector           │
│ • MyDroneDetector has_defense InputNormalization       │
│ • InputNormalization mitigates FGSM (effectiveness 0.3)│
│ • AdversarialPatch targets ObjectDetector              │
│ • ObjectDetector is_type_of CNN                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 2: Apply Class Hierarchy Reasoning
┌─────────────────────────────────────────────────────────┐
│ YOLOv5 is_type_of CNN                                  │
│ CNN is_type_of ObjectDetector (from ontology)          │
│ ─────────────────────────────                          │
│ INFERRED: YOLOv5 is_type_of ObjectDetector            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 3: Apply Rule 1 (Vulnerability Propagation)
┌─────────────────────────────────────────────────────────┐
│ FGSM targets CNN + YOLOv5 is_type_of CNN               │
│ ─────────────────────────────                          │
│ INFERRED: FGSM targets YOLOv5                          │
│                                                         │
│ MyDroneDetector uses YOLOv5 + FGSM targets YOLOv5     │
│ ─────────────────────────────                          │
│ INFERRED: MyDroneDetector vulnerable_to FGSM           │
│                                                         │
│ AdversarialPatch targets ObjectDetector                │
│ + YOLOv5 is_type_of ObjectDetector                     │
│ ─────────────────────────────                          │
│ INFERRED: AdversarialPatch targets YOLOv5              │
│ INFERRED: MyDroneDetector vulnerable_to AdversarialPatch│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 4: Apply Rule 2 (Variant Inheritance)
┌─────────────────────────────────────────────────────────┐
│ PGD variant_of FGSM + FGSM targets CNN                 │
│ ─────────────────────────────                          │
│ INFERRED: PGD targets CNN                              │
│ INFERRED: PGD targets YOLOv5                           │
│ INFERRED: MyDroneDetector vulnerable_to PGD            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 5: Apply Rule 3 (Defense Gap Detection)
┌─────────────────────────────────────────────────────────┐
│ MyDroneDetector vulnerable_to AdversarialPatch         │
│ MyDroneDetector has_defense InputNormalization         │
│ InputNormalization does NOT mitigate AdversarialPatch  │
│ ─────────────────────────────                          │
│ INFERRED: MyDroneDetector has_defense_gap AdversarialPatch│
│                                                         │
│ ⚠️  ALERT GENERATED: Critical vulnerability without    │
│     mitigation in Defense sector deployment            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 6: Apply Rule 4 (Defense Recommendation)
┌─────────────────────────────────────────────────────────┐
│ Query: What defenses mitigate AdversarialPatch?        │
│                                                         │
│ Results:                                                │
│   • DetectorEnsemble (effectiveness: 0.8, cost: high)  │
│   • AdversarialTraining (effectiveness: 0.6, cost: high)│
│   • SpatialSmoothing (effectiveness: 0.3, cost: low)   │
│                                                         │
│ RECOMMENDATION: Deploy DetectorEnsemble                │
│   Rationale: Highest effectiveness (0.8) against       │
│   AdversarialPatch for object detection models         │
└─────────────────────────────────────────────────────────┘

FINAL OUTPUT:
┌─────────────────────────────────────────────────────────┐
│ THREAT ASSESSMENT FOR: MyDroneDetector                 │
│ ═══════════════════════════════════════════════════════│
│                                                         │
│ VULNERABILITIES IDENTIFIED:                            │
│ ┌─────────────────┬──────────┬─────────────┬─────────┐ │
│ │ Attack          │ Severity │ Has Defense │ Status  │ │
│ ├─────────────────┼──────────┼─────────────┼─────────┤ │
│ │ FGSM            │ Medium   │ Yes (0.3)   │ Partial │ │
│ │ PGD             │ High     │ Yes (0.3)   │ Partial │ │
│ │ AdversarialPatch│ Critical │ No          │ ⚠️ GAP  │ │
│ └─────────────────┴──────────┴─────────────┴─────────┘ │
│                                                         │
│ RECOMMENDED ACTIONS:                                    │
│ 1. Deploy DetectorEnsemble (Priority: CRITICAL)        │
│ 2. Upgrade to AdversarialTraining (Priority: HIGH)     │
│ 3. Increase InputNormalization strength (Priority: MED)│
└─────────────────────────────────────────────────────────┘
```

---

# SECTION 3: THE DARK-G METHODOLOGY PIPELINE

## 3.1 Complete Pipeline Overview

```
DARK-G END-TO-END PIPELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  PHASE 1              PHASE 2              PHASE 3              PHASE 4    │
│  KNOWLEDGE            KNOWLEDGE            KNOWLEDGE            APPLICATION │
│  ACQUISITION          STRUCTURING          REASONING            & OUTPUT   │
│                                                                             │
│  ┌─────────┐         ┌─────────┐          ┌─────────┐         ┌─────────┐ │
│  │ Collect │         │ Extract │          │ Apply   │         │ Query   │ │
│  │ Sources │ ──────> │ & Map   │ ───────> │ Rules & │ ──────> │ Simulate│ │
│  │         │         │         │          │ Infer   │         │ Alert   │ │
│  └─────────┘         └─────────┘          └─────────┘         └─────────┘ │
│       │                   │                    │                    │      │
│       ▼                   ▼                    ▼                    ▼      │
│  • Papers             • Entity              • Forward            • Threat  │
│  • CVEs                 extraction            chaining             reports │
│  • GitHub             • Relation            • Rule               • Defense │
│  • Advisories           extraction            application          recs    │
│  • Incidents          • Ontology            • Consistency        • Alerts  │
│                         grounding             checking           • Compliance│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Phase 1: Knowledge Acquisition (Detailed)

### 3.2.1 Data Sources

```
SOURCE TYPES AND COLLECTION METHODS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────────┬─────────────────────────┬──────────────────────┐
│ Source Type        │ Examples                │ Collection Method    │
├────────────────────┼─────────────────────────┼──────────────────────┤
│ Research Papers    │ arXiv, IEEE, NeurIPS,   │ API polling (daily)  │
│                    │ USENIX, ACM CCS         │ RSS feeds            │
│                    │                         │ Keyword alerts       │
├────────────────────┼─────────────────────────┼──────────────────────┤
│ Security Advisories│ MITRE ATLAS             │ API integration      │
│                    │ NIST NVD                │ STIX/TAXII feeds     │
│                    │ ASD/ACSC advisories     │ Structured parsing   │
├────────────────────┼─────────────────────────┼──────────────────────┤
│ Code Repositories  │ GitHub (CleverHans,     │ GitHub API           │
│                    │ Foolbox, ART, AutoAttack│ README parsing       │
│                    │ advertorch)             │ Docstring extraction │
├────────────────────┼─────────────────────────┼──────────────────────┤
│ Incident Reports   │ AI Incident Database    │ Web scraping         │
│                    │ Vendor disclosures      │ Manual curation      │
│                    │ News articles           │ NER extraction       │
├────────────────────┼─────────────────────────┼──────────────────────┤
│ Expert Input       │ Security researchers    │ Web interface        │
│                    │ Domain experts          │ Validated additions  │
└────────────────────┴─────────────────────────┴──────────────────────┘
```

### 3.2.2 Continuous Monitoring Process

```
DAILY INGESTION WORKFLOW:
━━━━━━━━━━━━━━━━━━━━━━━━━

06:00 ─── Scheduled Job Starts ───────────────────────────────────────
          │
          ├──> Query arXiv API for new papers
          │    Keywords: "adversarial", "robust", "attack", "defense",
          │              "perturbation", "evasion", "poisoning"
          │    Filter: cs.LG, cs.CR, cs.CV categories
          │    
          ├──> Query GitHub API for repository updates
          │    Watch list: CleverHans, Foolbox, ART, AutoAttack, etc.
          │    Check: New releases, new attack implementations
          │    
          ├──> Poll MITRE ATLAS for new techniques
          │    
          └──> Check CVE feeds for ML-related vulnerabilities

07:00 ─── New Documents Queued ───────────────────────────────────────
          │
          └──> ~20-50 new documents per day typical

08:00 ─── LLM Processing Begins ──────────────────────────────────────
          │
          └──> Each document processed through extraction pipeline
               (Details in Phase 2)

12:00 ─── New Knowledge Integrated ───────────────────────────────────
          │
          ├──> New entities added to knowledge graph
          ├──> Relationships extracted and validated
          └──> Inference engine triggered for new implications

13:00 ─── Alert Generation ───────────────────────────────────────────
          │
          ├──> Check: Do new attacks affect registered systems?
          ├──> Check: Are there new defenses for known gaps?
          └──> Generate and send alerts to affected organisations
```

## 3.3 Phase 2: Knowledge Structuring (Detailed)

### 3.3.1 LLM-Based Entity Extraction

```
ENTITY EXTRACTION PIPELINE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT: Raw paper/document text

STEP 1: Document Classification
┌─────────────────────────────────────────────────────────────────────┐
│ PROMPT TO LLM:                                                      │
│ "Classify this document into one of:                                │
│  - attack_paper (introduces new attack)                             │
│  - defense_paper (introduces new defense)                           │
│  - evaluation_paper (compares attacks/defenses)                     │
│  - survey_paper (reviews field)                                     │
│  - incident_report (real-world attack)                              │
│                                                                     │
│ Also identify:                                                      │
│  - Primary domain (image, text, audio, tabular, graph)             │
│  - Model types discussed (CNN, transformer, RNN, etc.)             │
│                                                                     │
│ Document: [PAPER TEXT]"                                             │
│                                                                     │
│ OUTPUT: {type: "attack_paper", domain: "image", models: ["CNN"]}   │
└─────────────────────────────────────────────────────────────────────┘

STEP 2: Entity Extraction (Domain-Specific Prompts)
┌─────────────────────────────────────────────────────────────────────┐
│ FOR ATTACK PAPERS:                                                  │
│                                                                     │
│ PROMPT TO LLM:                                                      │
│ "Extract the following from this adversarial ML attack paper:       │
│                                                                     │
│ ATTACK INFORMATION:                                                 │
│  - Attack name (and any aliases)                                    │
│  - Attack type (evasion/poisoning/extraction/inference)            │
│  - Threat model (white-box/black-box/gray-box)                     │
│  - Perturbation type (Lp norm: L0, L1, L2, L-infinity)            │
│  - Typical perturbation budget (epsilon values tested)             │
│  - Computational complexity                                         │
│                                                                     │
│ TARGET INFORMATION:                                                 │
│  - Target task (classification/detection/segmentation/etc.)        │
│  - Target architectures tested (ResNet, VGG, YOLO, etc.)          │
│  - Datasets used for evaluation                                     │
│                                                                     │
│ EFFECTIVENESS:                                                      │
│  - Success rate reported                                            │
│  - Comparison to baseline attacks                                   │
│  - Defenses it bypasses (if mentioned)                             │
│                                                                     │
│ RELATIONSHIPS:                                                      │
│  - Is this a variant/improvement of another attack?                │
│  - What attacks does it compare against?                           │
│                                                                     │
│ Document: [PAPER TEXT]"                                             │
│                                                                     │
│ OUTPUT:                                                             │
│ {                                                                   │
│   "attack_name": "AutoAttack",                                     │
│   "aliases": ["AA"],                                                │
│   "attack_type": "evasion",                                        │
│   "threat_model": "white-box",                                     │
│   "perturbation": "L-infinity",                                    │
│   "epsilon_values": [8/255, 4/255],                                │
│   "target_task": "image_classification",                           │
│   "target_architectures": ["ResNet-50", "WideResNet", "VGG-16"],  │
│   "datasets": ["CIFAR-10", "CIFAR-100", "ImageNet"],              │
│   "success_rate": 0.99,                                            │
│   "improves_on": ["PGD", "FGSM"],                                  │
│   "bypasses": ["TRADES", "adversarial_training"],                  │
│   "source": "arXiv:2003.01690",                                    │
│   "year": 2020                                                      │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘

STEP 3: Ontology Grounding (Map to Standard Terms)
┌─────────────────────────────────────────────────────────────────────┐
│ PROBLEM: Papers use inconsistent terminology                        │
│                                                                     │
│ Paper says:          Ontology term:                                │
│ "ResNet-50"      →   amlo:ResNet50                                 │
│ "ResNet50"       →   amlo:ResNet50                                 │
│ "ResNet 50"      →   amlo:ResNet50                                 │
│                                                                     │
│ "L_inf"          →   amlo:LInfinity                                │
│ "L-infinity"     →   amlo:LInfinity                                │
│ "L∞"             →   amlo:LInfinity                                │
│                                                                     │
│ "PGD attack"     →   amlo:PGD                                      │
│ "Madry attack"   →   amlo:PGD                                      │
│ "Projected Gradient Descent" → amlo:PGD                            │
│                                                                     │
│ METHOD: Embedding similarity + alias dictionary + LLM verification │
└─────────────────────────────────────────────────────────────────────┘

STEP 4: Relation Extraction
┌─────────────────────────────────────────────────────────────────────┐
│ PROMPT TO LLM:                                                      │
│ "Given the extracted entities, identify relationships:              │
│                                                                     │
│ Entities: AutoAttack, PGD, FGSM, ResNet50, CIFAR-10               │
│                                                                     │
│ For each pair, determine if any of these relationships exist:      │
│  - variant_of (A is an improved version of B)                      │
│  - targets (Attack targets Architecture)                           │
│  - evaluated_on (Attack evaluated on Dataset)                      │
│  - bypasses (Attack bypasses Defense)                              │
│  - mitigates (Defense mitigates Attack)                            │
│                                                                     │
│ Output as list of (subject, relationship, object) triples."        │
│                                                                     │
│ OUTPUT:                                                             │
│ [                                                                   │
│   (AutoAttack, variant_of, PGD),                                   │
│   (AutoAttack, targets, ResNet50),                                 │
│   (AutoAttack, evaluated_on, CIFAR-10),                            │
│   (AutoAttack, bypasses, TRADES)                                   │
│ ]                                                                   │
└─────────────────────────────────────────────────────────────────────┘

STEP 5: Validation & Confidence Scoring
┌─────────────────────────────────────────────────────────────────────┐
│ CHECK 1: Schema Validation                                          │
│   - Does relationship match ontology definitions?                  │
│   - Is domain/range correct? (Attack can target Architecture,      │
│     but not Dataset)                                                │
│                                                                     │
│ CHECK 2: Consistency Validation                                     │
│   - Does this contradict existing facts?                           │
│   - Example: If we already have "X mitigates Y" with high          │
│     confidence, and new paper says "Y bypasses X", flag for review │
│                                                                     │
│ CHECK 3: Confidence Assignment                                      │
│   - Peer-reviewed paper: 0.9 base confidence                       │
│   - Preprint: 0.7 base confidence                                  │
│   - Blog post: 0.5 base confidence                                 │
│   - Multiple sources agree: +0.1 per source                        │
│                                                                     │
│ OUTPUT: Facts with confidence scores and provenance                │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3.2 Knowledge Graph Storage

```
GRAPH DATABASE STRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━

NODES (Entities):
┌─────────────────────────────────────────────────────────────────────┐
│ Attack Nodes:                                                       │
│   {id: "fgsm", type: "GradientAttack",                             │
│    properties: {name: "FGSM", year: 2015, perturbation: "Linf"}}  │
│                                                                     │
│ Defense Nodes:                                                      │
│   {id: "advtrain", type: "ProactiveDefense",                       │
│    properties: {name: "AdversarialTraining", overhead: "3x"}}      │
│                                                                     │
│ Architecture Nodes:                                                 │
│   {id: "cnn", type: "Architecture",                                │
│    properties: {name: "CNN", category: "convolutional"}}           │
│                                                                     │
│ Model Nodes (User-Registered):                                     │
│   {id: "my_model_001", type: "DeployedModel",                      │
│    properties: {name: "DroneDetector", org: "AusDefence"}}         │
└─────────────────────────────────────────────────────────────────────┘

EDGES (Relationships):
┌─────────────────────────────────────────────────────────────────────┐
│ (fgsm) ──[targets {conf: 0.95}]──> (cnn)                          │
│ (pgd) ──[variant_of {conf: 0.99}]──> (fgsm)                        │
│ (advtrain) ──[mitigates {effectiveness: 0.7}]──> (fgsm)           │
│ (my_model_001) ──[uses]──> (yolov5)                                │
│ (yolov5) ──[is_type_of]──> (cnn)                                   │
└─────────────────────────────────────────────────────────────────────┘

PROVENANCE (Attached to Every Fact):
┌─────────────────────────────────────────────────────────────────────┐
│ Every edge stores:                                                  │
│   - source: "arXiv:1412.6572" (paper ID)                           │
│   - extracted_date: "2025-01-15"                                   │
│   - confidence: 0.95                                                │
│   - extraction_method: "llm_gpt4"                                  │
│   - validated_by: ["human_expert", "cross_reference"]              │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.4 Phase 3: Knowledge Reasoning (Detailed)

### 3.4.1 Forward Chaining Inference

```
FORWARD CHAINING PROCESS:
━━━━━━━━━━━━━━━━━━━━━━━━━

Forward chaining starts with known facts and applies rules to derive new facts.
This continues until no new facts can be derived.

ITERATION 1: Apply Class Hierarchy Rules
┌─────────────────────────────────────────────────────────────────────┐
│ Rule: If X is_type_of Y, and Y is_type_of Z, then X is_type_of Z   │
│                                                                     │
│ Known: YOLOv5 is_type_of CNN                                       │
│ Known: CNN is_type_of ObjectDetector (from ontology)               │
│ Derived: YOLOv5 is_type_of ObjectDetector                          │
│                                                                     │
│ Known: ResNet50 is_type_of CNN                                     │
│ Derived: ResNet50 is_type_of ObjectDetector                        │
└─────────────────────────────────────────────────────────────────────┘

ITERATION 2: Apply Targeting Rules
┌─────────────────────────────────────────────────────────────────────┐
│ Rule: If Attack targets ArchClass, and Model is_type_of ArchClass, │
│       then Attack targets Model                                     │
│                                                                     │
│ Known: FGSM targets CNN                                            │
│ Known: YOLOv5 is_type_of CNN                                       │
│ Derived: FGSM targets YOLOv5                                       │
│                                                                     │
│ Known: AdversarialPatch targets ObjectDetector                     │
│ Derived (from Iter 1): YOLOv5 is_type_of ObjectDetector           │
│ Derived: AdversarialPatch targets YOLOv5                           │
└─────────────────────────────────────────────────────────────────────┘

ITERATION 3: Apply Variant Inheritance
┌─────────────────────────────────────────────────────────────────────┐
│ Rule: If A2 variant_of A1, and A1 targets Arch, then A2 targets Arch│
│                                                                     │
│ Known: PGD variant_of FGSM                                         │
│ Known: FGSM targets CNN                                            │
│ Derived: PGD targets CNN                                           │
│ Derived: PGD targets YOLOv5                                        │
│                                                                     │
│ Known: AutoAttack variant_of PGD                                   │
│ Derived: AutoAttack targets CNN                                    │
│ Derived: AutoAttack targets YOLOv5                                 │
└─────────────────────────────────────────────────────────────────────┘

ITERATION 4: Apply Vulnerability Rules
┌─────────────────────────────────────────────────────────────────────┐
│ Rule: If Model uses Arch, and Attack targets Arch,                 │
│       then Model potentially_vulnerable_to Attack                   │
│                                                                     │
│ Known: MyDroneDetector uses YOLOv5                                 │
│ Derived: FGSM targets YOLOv5                                       │
│ Derived: MyDroneDetector potentially_vulnerable_to FGSM            │
│                                                                     │
│ Derived: PGD targets YOLOv5                                        │
│ Derived: MyDroneDetector potentially_vulnerable_to PGD             │
│                                                                     │
│ Derived: AdversarialPatch targets YOLOv5                           │
│ Derived: MyDroneDetector potentially_vulnerable_to AdversarialPatch│
└─────────────────────────────────────────────────────────────────────┘

ITERATION 5: Apply Defense Gap Rules
┌─────────────────────────────────────────────────────────────────────┐
│ Rule: If Model vulnerable_to Attack, and Model has no Defense that │
│       mitigates Attack, then Model has_defense_gap Attack          │
│                                                                     │
│ Derived: MyDroneDetector vulnerable_to AdversarialPatch            │
│ Known: MyDroneDetector has_defense InputNormalization              │
│ Check: Does InputNormalization mitigate AdversarialPatch? NO       │
│ Derived: MyDroneDetector has_defense_gap AdversarialPatch          │
│                                                                     │
│ ⚠️ ALERT CONDITION MET                                              │
└─────────────────────────────────────────────────────────────────────┘

ITERATION 6: No New Facts Derivable → Stop
```

### 3.4.2 Query-Time Reasoning

```
SPARQL QUERY WITH INFERENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USER QUERY: "What attacks threaten my drone detector and what defenses work?"

SPARQL:
┌─────────────────────────────────────────────────────────────────────┐
│ PREFIX amlo: <http://dark-g.org/ontology#>                         │
│                                                                     │
│ SELECT ?attack ?attackType ?severity ?defense ?effectiveness       │
│ WHERE {                                                             │
│   # Find model's architecture                                       │
│   amlo:MyDroneDetector amlo:uses ?arch .                           │
│                                                                     │
│   # Find attacks targeting that architecture (including inferred)  │
│   ?attack amlo:targets ?arch .                                     │
│   ?attack rdf:type ?attackType .                                   │
│   ?attack amlo:severity ?severity .                                │
│                                                                     │
│   # Find defenses that mitigate each attack                        │
│   OPTIONAL {                                                        │
│     ?defense amlo:mitigates ?attack .                              │
│     ?defense amlo:effectivenessAgainst ?attack ?effectiveness .    │
│   }                                                                 │
│ }                                                                   │
│ ORDER BY DESC(?severity) DESC(?effectiveness)                      │
└─────────────────────────────────────────────────────────────────────┘

RESULT:
┌─────────────────────┬────────────────┬──────────┬──────────────────────┬───────────────┐
│ Attack              │ Type           │ Severity │ Defense              │ Effectiveness │
├─────────────────────┼────────────────┼──────────┼──────────────────────┼───────────────┤
│ AdversarialPatch    │ PhysicalAttack │ Critical │ DetectorEnsemble     │ 0.80          │
│ AdversarialPatch    │ PhysicalAttack │ Critical │ AdversarialTraining  │ 0.60          │
│ DAAC                │ PhysicalAttack │ Critical │ DetectorEnsemble     │ 0.65          │
│ PGD                 │ GradientAttack │ High     │ AdversarialTraining  │ 0.75          │
│ PGD                 │ GradientAttack │ High     │ InputNormalization   │ 0.40          │
│ FGSM                │ GradientAttack │ Medium   │ AdversarialTraining  │ 0.85          │
│ FGSM                │ GradientAttack │ Medium   │ InputNormalization   │ 0.50          │
└─────────────────────┴────────────────┴──────────┴──────────────────────┴───────────────┘
```

## 3.5 Phase 4: Application & Output (Detailed)

### 3.5.1 Threat Assessment Report Generation

```
AUTOMATED REPORT GENERATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT: Model registration + Knowledge graph + Inference results

OUTPUT: Structured threat assessment

┌─────────────────────────────────────────────────────────────────────┐
│            DARK-G THREAT ASSESSMENT REPORT                         │
│            ════════════════════════════════                         │
│                                                                     │
│ Model: MyDroneDetector                                             │
│ Organisation: Australian Defence Force                              │
│ Deployment Sector: Defence                                          │
│ Assessment Date: 2026-02-03                                         │
│ Knowledge Graph Version: 2026-02-03-r127                           │
│                                                                     │
│ ═══════════════════════════════════════════════════════════════════│
│ EXECUTIVE SUMMARY                                                   │
│ ───────────────────────────────────────────────────────────────────│
│ Total Vulnerabilities Identified: 6                                 │
│   • Critical: 2 (AdversarialPatch, DAAC)                           │
│   • High: 2 (PGD, AutoAttack)                                       │
│   • Medium: 2 (FGSM, BIM)                                          │
│                                                                     │
│ Defense Gaps: 2 attacks with no deployed mitigation                │
│ Recommended Actions: 3 (see below)                                  │
│                                                                     │
│ ═══════════════════════════════════════════════════════════════════│
│ VULNERABILITY DETAILS                                               │
│ ───────────────────────────────────────────────────────────────────│
│                                                                     │
│ 1. AdversarialPatch [CRITICAL] ⚠️ NO DEFENSE                        │
│    ─────────────────────────────────────────                        │
│    Type: Physical evasion attack                                   │
│    How it works: Printed pattern placed on target causes           │
│                  detection failure                                  │
│    Relevance: Your YOLOv5 architecture is vulnerable              │
│    Current defense: None deployed                                  │
│    Source: Brown et al. 2017, validated in TNO 2020 study         │
│                                                                     │
│    RECOMMENDED DEFENSE:                                             │
│    → DetectorEnsemble (effectiveness: 0.80)                        │
│      Use multiple detection models; patch optimized for one        │
│      often fails against others                                     │
│                                                                     │
│ 2. DAAC [CRITICAL] ⚠️ NO DEFENSE                                    │
│    ─────────────────────────────────────────                        │
│    Type: Dual-attribute adversarial camouflage                     │
│    How it works: Fools both AI detectors AND human observers       │
│    Relevance: Variant of AdversarialPatch, enhanced for            │
│               military applications                                 │
│    Current defense: None deployed                                  │
│    Source: Wang et al. 2021, Defence Technology journal            │
│                                                                     │
│    RECOMMENDED DEFENSE:                                             │
│    → DetectorEnsemble + AdversarialTraining                        │
│      Combined approach recommended due to attack sophistication    │
│                                                                     │
│ [Additional vulnerabilities continue...]                           │
│                                                                     │
│ ═══════════════════════════════════════════════════════════════════│
│ PRIORITISED RECOMMENDATIONS                                         │
│ ───────────────────────────────────────────────────────────────────│
│                                                                     │
│ Priority 1 [IMMEDIATE]: Deploy DetectorEnsemble                    │
│   Addresses: AdversarialPatch, DAAC                                │
│   Cost: High computational overhead                                 │
│   Effectiveness: 0.80 for AdversarialPatch, 0.65 for DAAC         │
│   Implementation: Use YOLOv5 + Faster-RCNN + SSD ensemble         │
│                                                                     │
│ Priority 2 [HIGH]: Implement AdversarialTraining                   │
│   Addresses: FGSM, PGD, AutoAttack, partial DAAC                   │
│   Cost: 3-5x training time                                         │
│   Effectiveness: 0.70-0.85 depending on attack                    │
│   Implementation: Retrain with PGD adversarial examples           │
│                                                                     │
│ Priority 3 [MEDIUM]: Upgrade InputNormalization                    │
│   Addresses: Improved FGSM, BIM mitigation                         │
│   Cost: Minimal                                                     │
│   Effectiveness: 0.50 (limited)                                    │
│                                                                     │
│ ═══════════════════════════════════════════════════════════════════│
│ COMPLIANCE MAPPING                                                  │
│ ───────────────────────────────────────────────────────────────────│
│                                                                     │
│ SOCI Act 2018:                                                     │
│   • Risk identified: AI system in critical infrastructure         │
│   • CIRMP Requirement: Document adversarial threats ✓              │
│   • Recommendation: Include this report in annual submission      │
│                                                                     │
│ Cyber Security Act 2024:                                           │
│   • Mandatory reporting: Would apply if breach occurs              │
│   • Proactive assessment: This report demonstrates due diligence  │
│                                                                     │
│ ═══════════════════════════════════════════════════════════════════│
│ PROVENANCE & CONFIDENCE                                             │
│ ───────────────────────────────────────────────────────────────────│
│                                                                     │
│ This assessment derived from:                                       │
│   • 47 peer-reviewed papers                                        │
│   • 12 MITRE ATLAS techniques                                      │
│   • 3 incident reports                                              │
│   • 156 inferred relationships                                     │
│                                                                     │
│ All facts traceable to sources. Click any finding for full        │
│ provenance chain and original publication links.                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.5.2 Continuous Monitoring Alerts

```
ALERT SYSTEM:
━━━━━━━━━━━━━

TRIGGER: New attack added to knowledge graph

PROCESS:
1. Extract new attack's target architectures
2. Query: Which registered models use these architectures?
3. Check: Do affected models have defenses that mitigate this attack?
4. If no defense: Generate CRITICAL alert
5. If partial defense: Generate WARNING alert

EXAMPLE ALERT:
┌─────────────────────────────────────────────────────────────────────┐
│ ⚠️  DARK-G SECURITY ALERT                                           │
│ ════════════════════════════════════════════════════════════════════│
│                                                                     │
│ NEW THREAT DETECTED                                                 │
│                                                                     │
│ Attack: ShipCamou (published 2026-01-28)                           │
│ Type: Physical adversarial patch for maritime vessel detection     │
│ Source: IEEE ICRA 2026                                              │
│                                                                     │
│ YOUR AFFECTED SYSTEMS:                                              │
│   • NavalReconSystem (uses YOLOv5) - NO DEFENSE                    │
│   • PortMonitor (uses Faster-RCNN) - PARTIAL DEFENSE               │
│                                                                     │
│ IMMEDIATE ACTIONS RECOMMENDED:                                      │
│   1. Review ShipCamou paper for attack details                     │
│   2. Consider DetectorEnsemble deployment                          │
│   3. Update threat model documentation                             │
│                                                                     │
│ [View Full Assessment] [Acknowledge] [Snooze 7 days]               │
└─────────────────────────────────────────────────────────────────────┘
```

---

# SECTION 4: DEFENCE CASE STUDY - ADVERSARIAL CAMOUFLAGE

## 4.1 Scenario Description

```
SCENARIO: Australian Defence Force Drone Surveillance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM: AI-powered drone for detecting military vehicles

DEPLOYMENT:
  • Drones patrol northern Australia
  • Onboard YOLOv5 detector identifies tanks, trucks, personnel
  • Real-time alerts sent to command center
  • Critical for early warning of incursions

THREAT:
  • Adversary wants to move military assets undetected
  • Traditional camouflage: Paint to blend with environment
  • NEW: Adversarial camouflage - patterns that fool AI

QUESTION:
  How do we assess and defend against adversarial camouflage threats?
```

## 4.2 How DARK-G Addresses This Scenario

### Step 1: Register the System

```
SYSTEM REGISTRATION:
━━━━━━━━━━━━━━━━━━━━

User inputs via web interface:
┌─────────────────────────────────────────────────────────────────────┐
│ System Name: ADF_DroneRecon_North                                   │
│ Model Architecture: YOLOv5-L                                        │
│ Task: Object Detection                                              │
│ Deployment Sector: Defence                                          │
│ Operational Environment: Aerial Surveillance                        │
│ Current Defenses:                                                   │
│   ☑ Input Normalization                                             │
│   ☑ Confidence Thresholding                                         │
│   ☐ Adversarial Training                                            │
│   ☐ Detector Ensemble                                               │
│ Threat Actors of Concern:                                           │
│   ☑ Nation State                                                    │
│   ☑ Military                                                        │
└─────────────────────────────────────────────────────────────────────┘

This creates nodes and edges in the knowledge graph:
  (ADF_DroneRecon_North) --uses--> (YOLOv5)
  (ADF_DroneRecon_North) --deployed_in--> (Defence_Sector)
  (ADF_DroneRecon_North) --has_defense--> (InputNormalization)
  (ADF_DroneRecon_North) --threat_actor--> (NationState)
```

### Step 2: Knowledge Graph Contains Relevant Attacks

```
EXISTING KNOWLEDGE (Extracted from Literature):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

From paper: "Adversarial Patch Camouflage against Aerial Detection" (TNO, 2020)
┌─────────────────────────────────────────────────────────────────────┐
│ Extracted entities:                                                 │
│   Attack: AdversarialPatch_Aerial                                  │
│   Targets: YOLO, Faster-RCNN                                       │
│   Context: Aerial/drone surveillance                               │
│   Effectiveness: 85% detection avoidance with 10% object coverage  │
│   Physical: Yes (printed patterns)                                 │
│                                                                     │
│ Stored triples:                                                     │
│   (AdversarialPatch_Aerial) --type--> (PhysicalEvasionAttack)     │
│   (AdversarialPatch_Aerial) --targets--> (YOLO)                    │
│   (AdversarialPatch_Aerial) --targets--> (ObjectDetector)          │
│   (AdversarialPatch_Aerial) --context--> (AerialSurveillance)     │
│   (AdversarialPatch_Aerial) --effectiveness--> (0.85)              │
│   (AdversarialPatch_Aerial) --source--> ("TNO_2020_aerial_patch") │
└─────────────────────────────────────────────────────────────────────┘

From paper: "DAAC: Dual Attribute Adversarial Camouflage" (2021)
┌─────────────────────────────────────────────────────────────────────┐
│ Extracted entities:                                                 │
│   Attack: DAAC                                                     │
│   Variant of: AdversarialPatch                                     │
│   Special property: Also fools human observers                     │
│   Military application: Soldier/vehicle camouflage                 │
│                                                                     │
│ Stored triples:                                                     │
│   (DAAC) --type--> (PhysicalEvasionAttack)                        │
│   (DAAC) --variant_of--> (AdversarialPatch)                        │
│   (DAAC) --fools_humans--> (true)                                  │
│   (DAAC) --targets--> (ObjectDetector)                             │
│   (DAAC) --application--> (MilitaryCamouflage)                     │
└─────────────────────────────────────────────────────────────────────┘

Defense knowledge:
┌─────────────────────────────────────────────────────────────────────┐
│ (DetectorEnsemble) --mitigates--> (AdversarialPatch)               │
│ (DetectorEnsemble) --effectiveness_vs--> (AdversarialPatch, 0.80)  │
│ (DetectorEnsemble) --mitigates--> (DAAC)                           │
│ (DetectorEnsemble) --effectiveness_vs--> (DAAC, 0.65)              │
│                                                                     │
│ (AdversarialTraining) --mitigates--> (AdversarialPatch)            │
│ (AdversarialTraining) --effectiveness_vs--> (AdversarialPatch, 0.60)│
│                                                                     │
│ (InputNormalization) --mitigates--> (FGSM)                         │
│ (InputNormalization) --NOT_mitigates--> (AdversarialPatch)         │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 3: Inference Engine Runs

```
INFERENCE TRACE FOR ADF_DroneRecon_North:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GIVEN FACTS:
  • ADF_DroneRecon_North uses YOLOv5
  • YOLOv5 is_type_of YOLO
  • YOLO is_type_of ObjectDetector
  • AdversarialPatch_Aerial targets YOLO
  • DAAC variant_of AdversarialPatch
  • AdversarialPatch targets ObjectDetector
  • ADF_DroneRecon_North has_defense InputNormalization
  • InputNormalization NOT_mitigates AdversarialPatch

INFERENCE STEP 1: Architecture chain
  YOLOv5 is_type_of YOLO is_type_of ObjectDetector
  → YOLOv5 is_type_of ObjectDetector ✓

INFERENCE STEP 2: Attack targeting
  AdversarialPatch_Aerial targets YOLO
  + YOLOv5 is_type_of YOLO
  → AdversarialPatch_Aerial targets YOLOv5 ✓
  
  AdversarialPatch targets ObjectDetector
  + YOLOv5 is_type_of ObjectDetector
  → AdversarialPatch targets YOLOv5 ✓
  
  DAAC variant_of AdversarialPatch
  + AdversarialPatch targets ObjectDetector
  → DAAC targets ObjectDetector
  → DAAC targets YOLOv5 ✓

INFERENCE STEP 3: System vulnerability
  ADF_DroneRecon_North uses YOLOv5
  + AdversarialPatch_Aerial targets YOLOv5
  → ADF_DroneRecon_North vulnerable_to AdversarialPatch_Aerial ✓
  
  + DAAC targets YOLOv5
  → ADF_DroneRecon_North vulnerable_to DAAC ✓

INFERENCE STEP 4: Defense gap check
  ADF_DroneRecon_North vulnerable_to AdversarialPatch_Aerial
  + ADF_DroneRecon_North has_defense InputNormalization
  + InputNormalization NOT_mitigates AdversarialPatch
  → ADF_DroneRecon_North has_defense_gap AdversarialPatch_Aerial ⚠️
  
  ADF_DroneRecon_North vulnerable_to DAAC
  + No defense mitigates DAAC
  → ADF_DroneRecon_North has_defense_gap DAAC ⚠️

INFERENCE STEP 5: Context-specific severity
  ADF_DroneRecon_North deployed_in Defence_Sector
  + ADF_DroneRecon_North threat_actor NationState
  + DAAC application MilitaryCamouflage
  → DAAC threat_relevance CRITICAL (military context + state adversary) ⚠️

INFERENCE COMPLETE.
```

### Step 4: Generated Output

```
OUTPUT: Threat Assessment for ADF_DroneRecon_North
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────┐
│ ADVERSARIAL CAMOUFLAGE THREAT ASSESSMENT                           │
│ ════════════════════════════════════════════════════════════════════│
│                                                                     │
│ System: ADF_DroneRecon_North                                        │
│ Assessment Date: 2026-02-03                                         │
│                                                                     │
│ CRITICAL FINDINGS:                                                  │
│ ──────────────────                                                  │
│                                                                     │
│ Your drone reconnaissance system is VULNERABLE to adversarial      │
│ camouflage attacks that could allow adversary assets to evade      │
│ detection. These attacks have been demonstrated in peer-reviewed   │
│ military research.                                                  │
│                                                                     │
│ SPECIFIC THREATS:                                                   │
│                                                                     │
│ 1. Adversarial Patch (Aerial) - CRITICAL, NO DEFENSE               │
│    ────────────────────────────────────────────────                 │
│    What: Printed pattern (10-30% of object size) placed on         │
│          military vehicles causes YOLOv5 to miss detection         │
│    Evidence: TNO Netherlands demonstrated 85% evasion rate         │
│    Your exposure: Direct - uses exact architecture tested          │
│    Current defense: None effective                                 │
│                                                                     │
│ 2. DAAC (Dual Attribute Adversarial Camouflage) - CRITICAL         │
│    ────────────────────────────────────────────────                 │
│    What: Advanced camouflage that fools BOTH AI and humans         │
│    Evidence: Published in Defence Technology journal 2021          │
│    Military relevance: Explicitly designed for soldier/vehicle     │
│                        concealment from drone surveillance          │
│    Your exposure: High - variant of attacks targeting your arch    │
│    Current defense: None effective                                 │
│                                                                     │
│ WHY YOUR CURRENT DEFENSES DON'T WORK:                              │
│ ──────────────────────────────────────                              │
│                                                                     │
│ Input Normalization: ✗ Designed for digital perturbations          │
│                        Physical patches survive preprocessing      │
│                                                                     │
│ Confidence Thresholding: ✗ Patches cause complete detection        │
│                            failure, not low-confidence detections  │
│                                                                     │
│ RECOMMENDED DEFENSES (Priority Order):                             │
│ ──────────────────────────────────────                              │
│                                                                     │
│ 1. DETECTOR ENSEMBLE [Effectiveness: 80%]                          │
│    Deploy multiple detection models (YOLOv5 + Faster-RCNN + SSD)  │
│    Patches optimized for one detector often fail on others        │
│    Implementation time: 2-4 weeks                                   │
│    Computational cost: 3x inference time                           │
│                                                                     │
│ 2. ADVERSARIAL TRAINING [Effectiveness: 60%]                       │
│    Retrain YOLOv5 with adversarial patch examples                  │
│    Improves robustness but not complete defense                    │
│    Implementation time: 4-6 weeks (requires retraining)           │
│                                                                     │
│ 3. PATCH DETECTION MODULE [Experimental]                           │
│    Secondary model to detect presence of adversarial patterns      │
│    Flags suspicious regions for human review                       │
│    Research stage - contact DSTG for collaboration                │
│                                                                     │
│ PROVENANCE:                                                         │
│ ───────────                                                         │
│ This assessment based on:                                          │
│   • Adhikari et al. "Adversarial Patch Camouflage against Aerial   │
│     Detection" (TNO, 2020)                                          │
│   • Wang et al. "DAAC" (Defence Technology, 2021)                  │
│   • 12 additional peer-reviewed sources                            │
│   • MITRE ATLAS technique AML.T0015                                │
│                                                                     │
│ All findings traceable. [View Full Provenance Chain]               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

# SECTION 5: RESEARCH CONTRIBUTIONS AND METHODOLOGY INNOVATION

## 5.1 Novel Contributions of DARK-G

```
RESEARCH CONTRIBUTIONS:
━━━━━━━━━━━━━━━━━━━━━━━

1. FIRST SEMANTIC ONTOLOGY FOR ADVERSARIAL ML
   ────────────────────────────────────────────
   • Formal OWL ontology covering attacks, defenses, architectures
   • Alignment with MITRE ATLAS, NIST AI RMF, OWASP ML Top 10
   • Inference rules for automated threat assessment
   
2. LLM-POWERED KNOWLEDGE EXTRACTION PIPELINE
   ────────────────────────────────────────────
   • Domain-specific prompts for adversarial ML papers
   • Ontology-grounded entity resolution
   • Confidence scoring and provenance tracking
   
3. CONTEXT-AWARE VULNERABILITY REASONING
   ────────────────────────────────────────────
   • Sector-specific threat relevance assessment
   • Deployment context influences severity ratings
   • Regulatory compliance mapping
   
4. CONTINUOUS THREAT MONITORING FRAMEWORK
   ────────────────────────────────────────────
   • Real-time ingestion from publication sources
   • Automated alert generation for new threats
   • Defense gap analysis at scale
```

## 5.2 Methodology Summary

```
DARK-G METHODOLOGY - ONE PAGE SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEM: 
  Adversarial ML knowledge is fragmented, static, and cannot scale.
  
SOLUTION:
  Semantic Knowledge Graph with automated reasoning.

HOW IT WORKS:

  1. ACQUIRE: Continuously collect papers, CVEs, code, incidents
              ↓
  2. EXTRACT: LLM extracts attacks, defenses, relationships
              ↓
  3. STRUCTURE: Map to formal ontology with defined semantics
              ↓
  4. STORE: Knowledge graph database with provenance
              ↓
  5. REASON: Forward-chaining inference derives new knowledge
              ↓
  6. QUERY: Context-aware threat assessment for specific systems
              ↓
  7. ALERT: Continuous monitoring for new relevant threats

KEY INNOVATION:
  The semantic layer enables REASONING - the system can figure out
  "Attack X affects Architecture Y, your model uses Y, therefore
  your model is vulnerable" WITHOUT this being explicitly stored.
  
  This is what databases and LLMs cannot do.

OUTCOMES:
  • Threat awareness: Months → Days
  • Assessment effort: Weeks → Minutes  
  • Defense gaps: Hidden → Visible
  • Knowledge decay: Rapid → Continuously updated
  • Reasoning: Manual → Automated
```

---

# Summary

This document has explained:

1. **The Problem**: Why current approaches (manual review, databases, LLMs) fail at adversarial ML threat management

2. **The Solution**: How semantic knowledge graphs enable automated reasoning about threats through:
   - Ontology (schema + rules)
   - Facts (triples)
   - Inference (deriving new knowledge)

3. **The Pipeline**: Complete methodology from knowledge acquisition through to threat reports:
   - Phase 1: Continuous ingestion from papers, CVEs, code
   - Phase 2: LLM-powered extraction with ontology grounding
   - Phase 3: Forward-chaining inference with rules
   - Phase 4: Threat reports, recommendations, alerts

4. **Case Study**: Adversarial camouflage threats to drone surveillance - showing exactly how DARK-G identifies vulnerabilities and recommends defenses

5. **Research Contributions**: Novel ontology, extraction pipeline, reasoning framework

The key insight: **Semantic knowledge graphs don't just store information - they understand and reason about it.** This is what makes automated, scalable, context-aware threat assessment possible.
