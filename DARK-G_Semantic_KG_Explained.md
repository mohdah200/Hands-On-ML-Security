# DARK-G: The Semantic Knowledge Graph Solution for Adversarial ML
## A Complete Technical Explanation

---

# Part 1: Understanding the Problem

## 1.1 The Current State: Why Organisations Are Failing at Adversarial ML Defense

Every day, researchers publish new ways to fool machine learning models:
- **Monday**: A paper shows how to make a stop sign invisible to autonomous vehicles
- **Tuesday**: A new attack extracts training data from language models  
- **Wednesday**: Researchers demonstrate how to poison datasets to create backdoors
- **Thursday**: A technique bypasses all known defenses against image classifiers
- **Friday**: An attack transfers from one model to another without any knowledge of the target

**The fundamental problem**: There is NO systematic way to:
1. Track all these attacks as they emerge
2. Understand which attacks apply to YOUR specific models
3. Know which defenses work against which attacks
4. Assess your risk continuously as the threat landscape changes

### Current Approaches and Why They Fail

| Approach | What It Does | Why It Fails |
|----------|--------------|--------------|
| **Literature Reviews** | Manually read papers | Outdated immediately; cannot scale |
| **Vulnerability Databases** | List known issues | No relationships; no reasoning; no context |
| **Penetration Testing Tools** | Test for known attacks | No knowledge of new attacks; no prioritization |
| **Expert Consultants** | Provide point-in-time assessment | Expensive; cannot monitor continuously |
| **LLM Assistants** | Answer questions about attacks | Hallucinate; no grounding; no traceability |

**The gap**: We need a system that KNOWS about attacks, UNDERSTANDS how they relate to models and defenses, REASONS about applicability, and UPDATES continuously.

---

# Part 2: The Solution - Semantic Knowledge Graphs Explained

## 2.1 What is a Knowledge Graph?

A knowledge graph stores information as a network of **entities** (things) connected by **relationships** (how things relate).

**Example - Simple Facts:**
```
[FGSM Attack] --targets--> [Image Classifier]
[FGSM Attack] --publishedIn--> [ICLR 2015]
[FGSM Attack] --createdBy--> [Goodfellow et al.]
[Adversarial Training] --mitigates--> [FGSM Attack]
```

This is better than a database because relationships are FIRST-CLASS citizens, not hidden in foreign keys.

## 2.2 What Makes It "Semantic"?

A **semantic** knowledge graph adds MEANING through an **ontology** - a formal definition of:
- What types of entities exist (Attack, Defense, Model, Dataset...)
- What properties they can have (perturbation_budget, success_rate...)
- What relationships are valid (Attack can "target" Model, but not "target" Dataset)
- What rules govern inference (if A mitigates B, and C is a variant of B, then A may partially mitigate C)

### The Power of Semantics: A Concrete Example

**Without Semantics (Regular Database):**
```sql
SELECT * FROM attacks WHERE name = 'FGSM';
-- Returns: {name: 'FGSM', type: 'evasion', year: 2015}
-- That's ALL you get. No connections. No meaning.
```

**With Semantics (Knowledge Graph):**
```sparql
SELECT ?attack ?defense ?effectiveness ?model_type
WHERE {
  ?attack rdf:type aml:EvasionAttack .
  ?attack aml:targets ?architecture .
  ?defense aml:mitigates ?attack .
  ?defense aml:effectivenessAgainst ?attack ?effectiveness .
  ?myModel aml:usesArchitecture ?architecture .
}
```
This query finds: All evasion attacks that could affect my model, with their defenses and effectiveness ratings - **even if I never explicitly stored "this attack affects my model"**.

## 2.3 The Three Pillars of Semantic Knowledge Graphs

### Pillar 1: RDF (Resource Description Framework) - The Data Model

Everything is stored as **triples**: (Subject, Predicate, Object)

```turtle
# Example RDF triples for adversarial ML
aml:FGSM rdf:type aml:EvasionAttack .
aml:FGSM aml:perturbationNorm "L-infinity" .
aml:FGSM aml:typicalEpsilon "0.03"^^xsd:float .
aml:FGSM aml:targets aml:CNN .
aml:FGSM aml:targets aml:ResNet .
aml:AdversarialTraining aml:mitigates aml:FGSM .
aml:AdversarialTraining aml:computationalOverhead "3x training time" .
```

### Pillar 2: OWL (Web Ontology Language) - The Schema + Rules

OWL defines the **ontology** - the formal structure of knowledge:

```turtle
# Class hierarchy
aml:Attack rdf:type owl:Class .
aml:EvasionAttack rdfs:subClassOf aml:Attack .
aml:PoisoningAttack rdfs:subClassOf aml:Attack .
aml:GradientBasedAttack rdfs:subClassOf aml:EvasionAttack .
aml:FGSM rdfs:subClassOf aml:GradientBasedAttack .
aml:PGD rdfs:subClassOf aml:GradientBasedAttack .

# Property definitions
aml:targets rdf:type owl:ObjectProperty .
aml:targets rdfs:domain aml:Attack .
aml:targets rdfs:range aml:Architecture .

aml:mitigates rdf:type owl:ObjectProperty .
aml:mitigates rdfs:domain aml:Defense .
aml:mitigates rdfs:range aml:Attack .

# Inference rule: transitive vulnerability
aml:variantOf rdf:type owl:TransitiveProperty .
# If FGSM targets CNN, and PGD is a variant of FGSM, 
# then PGD also targets CNN (inferred automatically!)
```

### Pillar 3: SPARQL - The Query Language

SPARQL queries can traverse relationships and leverage inference:

```sparql
# Find all attacks that might affect my YOLOv5 model
PREFIX aml: <http://dark-g.org/ontology#>

SELECT ?attack ?severity ?knownDefense ?defenseEffectiveness
WHERE {
  # My model uses a specific architecture
  aml:MyYOLOv5 aml:usesArchitecture ?arch .
  
  # Find attacks targeting that architecture (or parent architectures)
  ?attack aml:targets ?arch .
  ?attack aml:severity ?severity .
  
  # Find defenses that mitigate those attacks
  OPTIONAL {
    ?knownDefense aml:mitigates ?attack .
    ?knownDefense aml:effectivenessAgainst ?attack ?defenseEffectiveness .
  }
}
ORDER BY DESC(?severity)
```

## 2.4 The Magic: Automated Reasoning

The semantic layer enables **inference** - deriving NEW facts from existing ones.

### Example: Vulnerability Propagation

**Explicit facts we stored:**
```turtle
aml:FGSM aml:targets aml:CNN .
aml:ResNet50 rdf:type aml:CNN .
aml:MyModel aml:isInstanceOf aml:ResNet50 .
```

**Ontology rule:**
```turtle
# If an attack targets an architecture class,
# it targets all instances of that class
aml:targets rdfs:range aml:Architecture .
aml:isInstanceOf rdfs:subPropertyOf rdf:type .
```

**Inferred fact (automatic!):**
```turtle
aml:FGSM aml:potentiallyAffects aml:MyModel .
```

We **never explicitly stated** that FGSM affects MyModel. The reasoner **derived it** from the class hierarchy and property definitions.

### Example: Defense Gap Analysis

**Explicit facts:**
```turtle
aml:MyModel aml:hasDefense aml:InputNormalization .
aml:InputNormalization aml:mitigates aml:FGSM .
aml:InputNormalization aml:mitigates aml:BIM .
aml:AdversarialPatch aml:targets aml:CNN .
```

**Query to find unmitigated threats:**
```sparql
SELECT ?attack ?severity
WHERE {
  aml:MyModel aml:usesArchitecture ?arch .
  ?attack aml:targets ?arch .
  ?attack aml:severity ?severity .
  
  # Find attacks with NO defense currently deployed
  FILTER NOT EXISTS {
    aml:MyModel aml:hasDefense ?defense .
    ?defense aml:mitigates ?attack .
  }
}
```

**Result:** "AdversarialPatch (severity: HIGH) - NO DEFENSE DEPLOYED"

---

# Part 3: The DARK-G Architecture

## 3.1 System Overview

DARK-G transforms the semantic knowledge graph principles into a complete adversarial ML robustness platform:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DARK-G Architecture                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ Knowledge       │    │ Ontology        │    │ Inference       │     │
│  │ Acquisition     │───▶│ Layer           │───▶│ Engine          │     │
│  │                 │    │                 │    │                 │     │
│  │ • Papers        │    │ • Attack Types  │    │ • Rule-based    │     │
│  │ • CVEs/Advisories│   │ • Defense Types │    │ • ML-enhanced   │     │
│  │ • GitHub repos  │    │ • Model Archs   │    │ • Link prediction│    │
│  │ • Incident reports│  │ • Relationships │    │ • Trend detection│    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│           │                     │                      │               │
│           ▼                     ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Knowledge Graph Database                      │   │
│  │                                                                  │   │
│  │   (Attack)──targets──▶(Architecture)◀──uses──(Model)            │   │
│  │      │                      │                   │                │   │
│  │      │                      │                   │                │   │
│  │   mitigatedBy            affectedBy         deployedIn          │   │
│  │      │                      │                   │                │   │
│  │      ▼                      ▼                   ▼                │   │
│  │  (Defense)              (Sector)           (Context)            │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                     │                      │               │
│           ▼                     ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ Query           │    │ Simulation      │    │ Alert &         │     │
│  │ Interface       │    │ Platform        │    │ Reporting       │     │
│  │                 │    │                 │    │                 │     │
│  │ "What threatens │    │ Automated       │    │ New threat      │     │
│  │  my model?"     │    │ red-teaming     │    │ notifications   │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Detailed Component Design

### Component 1: Knowledge Acquisition Layer

**Purpose:** Continuously extract structured knowledge from unstructured sources

**How it works:**

```
┌──────────────────────────────────────────────────────────────────┐
│                  LLM-Powered Extraction Pipeline                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Document Ingestion                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
│  │ arXiv API  │  │ GitHub API │  │ CVE Feed   │                 │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                 │
│        └───────────────┼───────────────┘                        │
│                        ▼                                         │
│  Step 2: Document Classification                                 │
│  ┌─────────────────────────────────────────┐                    │
│  │ LLM Classifier                          │                    │
│  │ Input: "Towards Evaluating the..."      │                    │
│  │ Output: {type: "attack_paper",          │                    │
│  │          domain: "image_classification"}│                    │
│  └─────────────────┬───────────────────────┘                    │
│                    ▼                                             │
│  Step 3: Entity Extraction (Domain-Specific Prompts)            │
│  ┌─────────────────────────────────────────┐                    │
│  │ Prompt: "Extract from this paper:       │                    │
│  │  - Attack name and aliases              │                    │
│  │  - Target model architectures           │                    │
│  │  - Perturbation type (Lp norm)          │                    │
│  │  - Success metrics reported             │                    │
│  │  - Compared baselines..."               │                    │
│  │                                          │                    │
│  │ Output: {                                │                    │
│  │   attack: "AutoAttack",                 │                    │
│  │   targets: ["ResNet", "VGG", "Inception"],│                   │
│  │   perturbation: "L-infinity",            │                    │
│  │   epsilon: [0.03, 0.1, 0.3],            │                    │
│  │   success_rate: 0.99,                   │                    │
│  │   baselines: ["FGSM", "PGD", "C&W"]     │                    │
│  │ }                                        │                    │
│  └─────────────────┬───────────────────────┘                    │
│                    ▼                                             │
│  Step 4: Relation Extraction                                     │
│  ┌─────────────────────────────────────────┐                    │
│  │ Prompt: "What relationships exist?       │                    │
│  │  - Does this attack improve on others?  │                    │
│  │  - What defenses does it bypass?        │                    │
│  │  - What defenses might mitigate it?"    │                    │
│  │                                          │                    │
│  │ Output: [                                │                    │
│  │   (AutoAttack, improves_on, PGD),       │                    │
│  │   (AutoAttack, bypasses, InputTransform),│                   │
│  │   (AutoAttack, partiallyMitigatedBy,    │                    │
│  │    AdversarialTraining)                 │                    │
│  │ ]                                        │                    │
│  └─────────────────┬───────────────────────┘                    │
│                    ▼                                             │
│  Step 5: Ontology Grounding & Validation                        │
│  ┌─────────────────────────────────────────┐                    │
│  │ • Map extracted entities to ontology    │                    │
│  │ • Resolve aliases (PGD = Projected      │                    │
│  │   Gradient Descent = Madry Attack)      │                    │
│  │ • Validate relationships against schema │                    │
│  │ • Compute confidence scores             │                    │
│  │ • Store provenance (source, date, etc.) │                    │
│  └─────────────────────────────────────────┘                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Component 2: The Adversarial ML Ontology (AMLO)

**Purpose:** Define the formal structure of adversarial ML knowledge

```turtle
# ═══════════════════════════════════════════════════════════════════
# ADVERSARIAL ML ONTOLOGY (AMLO) - Core Classes
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# ATTACK TAXONOMY (aligned with MITRE ATLAS)
# ─────────────────────────────────────────────────────────────────
amlo:Attack rdf:type owl:Class ;
    rdfs:comment "Any technique to compromise ML system integrity" .

amlo:EvasionAttack rdfs:subClassOf amlo:Attack ;
    rdfs:comment "Attacks at inference time to cause misclassification" .

amlo:PoisoningAttack rdfs:subClassOf amlo:Attack ;
    rdfs:comment "Attacks that corrupt training data or process" .

amlo:ExtractionAttack rdfs:subClassOf amlo:Attack ;
    rdfs:comment "Attacks to steal model or training data" .

amlo:InferenceAttack rdfs:subClassOf amlo:Attack ;
    rdfs:comment "Attacks to infer sensitive information" .

# Evasion Attack Subtypes
amlo:GradientBasedAttack rdfs:subClassOf amlo:EvasionAttack .
amlo:ScoreBasedAttack rdfs:subClassOf amlo:EvasionAttack .
amlo:DecisionBasedAttack rdfs:subClassOf amlo:EvasionAttack .
amlo:PhysicalAttack rdfs:subClassOf amlo:EvasionAttack .

# Specific Attacks
amlo:FGSM rdfs:subClassOf amlo:GradientBasedAttack ;
    amlo:firstPublished "2015"^^xsd:gYear ;
    amlo:perturbationNorm amlo:LInfinity .

amlo:PGD rdfs:subClassOf amlo:GradientBasedAttack ;
    amlo:improves amlo:FGSM ;
    amlo:firstPublished "2018"^^xsd:gYear .

amlo:AdversarialPatch rdfs:subClassOf amlo:PhysicalAttack ;
    amlo:requiresPhysicalAccess "true"^^xsd:boolean .

# ─────────────────────────────────────────────────────────────────
# DEFENSE TAXONOMY (aligned with NIST AI RMF)
# ─────────────────────────────────────────────────────────────────
amlo:Defense rdf:type owl:Class .

amlo:ProactiveDefense rdfs:subClassOf amlo:Defense ;
    rdfs:comment "Defenses applied during training" .

amlo:ReactiveDefense rdfs:subClassOf amlo:Defense ;
    rdfs:comment "Defenses applied at inference time" .

amlo:AdversarialTraining rdfs:subClassOf amlo:ProactiveDefense ;
    amlo:computationalOverhead "2-10x training cost" .

amlo:InputPreprocessing rdfs:subClassOf amlo:ReactiveDefense ;
    amlo:computationalOverhead "minimal" .

amlo:CertifiedDefense rdfs:subClassOf amlo:ProactiveDefense ;
    amlo:providesGuarantees "true"^^xsd:boolean .

# ─────────────────────────────────────────────────────────────────
# MODEL & ARCHITECTURE TAXONOMY
# ─────────────────────────────────────────────────────────────────
amlo:Architecture rdf:type owl:Class .

amlo:CNN rdfs:subClassOf amlo:Architecture .
amlo:Transformer rdfs:subClassOf amlo:Architecture .
amlo:RNN rdfs:subClassOf amlo:Architecture .

amlo:ResNet rdfs:subClassOf amlo:CNN .
amlo:VGG rdfs:subClassOf amlo:CNN .
amlo:YOLO rdfs:subClassOf amlo:CNN .
amlo:ViT rdfs:subClassOf amlo:Transformer .

# ─────────────────────────────────────────────────────────────────
# CONTEXT TAXONOMY
# ─────────────────────────────────────────────────────────────────
amlo:Sector rdf:type owl:Class .
amlo:Healthcare rdfs:subClassOf amlo:Sector .
amlo:Finance rdfs:subClassOf amlo:Sector .
amlo:Defense rdfs:subClassOf amlo:Sector .
amlo:CriticalInfrastructure rdfs:subClassOf amlo:Sector .

amlo:Regulation rdf:type owl:Class .
amlo:SOCIAct rdf:type amlo:Regulation ;
    amlo:appliesToSector amlo:CriticalInfrastructure .

# ═══════════════════════════════════════════════════════════════════
# KEY RELATIONSHIPS (Object Properties)
# ═══════════════════════════════════════════════════════════════════

amlo:targets rdf:type owl:ObjectProperty ;
    rdfs:domain amlo:Attack ;
    rdfs:range amlo:Architecture .

amlo:mitigates rdf:type owl:ObjectProperty ;
    rdfs:domain amlo:Defense ;
    rdfs:range amlo:Attack .

amlo:bypasses rdf:type owl:ObjectProperty ;
    rdfs:domain amlo:Attack ;
    rdfs:range amlo:Defense .

amlo:variantOf rdf:type owl:ObjectProperty, owl:TransitiveProperty ;
    rdfs:domain amlo:Attack ;
    rdfs:range amlo:Attack .

amlo:usesArchitecture rdf:type owl:ObjectProperty ;
    rdfs:domain amlo:Model ;
    rdfs:range amlo:Architecture .

amlo:deployedIn rdf:type owl:ObjectProperty ;
    rdfs:domain amlo:Model ;
    rdfs:range amlo:Sector .

# ═══════════════════════════════════════════════════════════════════
# INFERENCE RULES (SWRL)
# ═══════════════════════════════════════════════════════════════════

# Rule 1: Vulnerability Propagation
# If attack A targets architecture X, and model M uses architecture X,
# then model M is potentially vulnerable to attack A
Attack(?a) ∧ targets(?a, ?arch) ∧ Model(?m) ∧ usesArchitecture(?m, ?arch) 
    → potentiallyVulnerableTo(?m, ?a)

# Rule 2: Variant Vulnerability
# If attack A2 is a variant of attack A1, and A1 targets architecture X,
# then A2 also targets architecture X
variantOf(?a2, ?a1) ∧ targets(?a1, ?arch) → targets(?a2, ?arch)

# Rule 3: Defense Gap Detection
# If model M is potentially vulnerable to attack A, 
# and M has no defense that mitigates A, then M has a defense gap
potentiallyVulnerableTo(?m, ?a) ∧ ¬(hasDefense(?m, ?d) ∧ mitigates(?d, ?a))
    → hasDefenseGap(?m, ?a)

# Rule 4: Regulatory Implications
# If model M is deployed in sector S, and regulation R applies to S,
# and attack A could violate R's requirements, then A has compliance risk for M
deployedIn(?m, ?s) ∧ appliesToSector(?r, ?s) ∧ violatesRequirement(?a, ?r)
    → hasComplianceRisk(?m, ?a, ?r)
```

### Component 3: The Inference Engine

**Purpose:** Automatically derive new knowledge from explicit facts + rules

```
┌──────────────────────────────────────────────────────────────────┐
│                     Inference Engine                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Explicit Facts                                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ • FGSM targets CNN                                      │     │
│  │ • PGD is variant of FGSM                                │     │
│  │ • AutoAttack is variant of PGD                          │     │
│  │ • MyDroneDetector uses YOLOv5                           │     │
│  │ • YOLOv5 is a CNN                                       │     │
│  │ • MyDroneDetector deployed in Defense sector            │     │
│  │ • MyDroneDetector has defense: InputNormalization       │     │
│  │ • InputNormalization mitigates FGSM                     │     │
│  │ • AdversarialPatch bypasses InputNormalization          │     │
│  └─────────────────────────────────────────────────────────┘     │
│                           │                                      │
│                           ▼                                      │
│  REASONING (Forward Chaining)                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                                                         │     │
│  │  Step 1: Class Hierarchy Inference                      │     │
│  │    YOLOv5 is CNN → YOLOv5 inherits all CNN properties  │     │
│  │                                                         │     │
│  │  Step 2: Transitive Property Inference                  │     │
│  │    PGD variant of FGSM + AutoAttack variant of PGD      │     │
│  │    → AutoAttack variant of FGSM (transitivity)          │     │
│  │                                                         │     │
│  │  Step 3: Rule Application                               │     │
│  │    FGSM targets CNN + YOLOv5 is CNN                     │     │
│  │    → FGSM targets YOLOv5 (class membership)             │     │
│  │                                                         │     │
│  │    FGSM targets YOLOv5 + MyDroneDetector uses YOLOv5    │     │
│  │    → MyDroneDetector potentiallyVulnerableTo FGSM       │     │
│  │                                                         │     │
│  │    AutoAttack variant of FGSM + FGSM targets CNN        │     │
│  │    → AutoAttack targets CNN                             │     │
│  │    → MyDroneDetector potentiallyVulnerableTo AutoAttack │     │
│  │                                                         │     │
│  │  Step 4: Defense Gap Analysis                           │     │
│  │    MyDroneDetector vulnerable to AdversarialPatch       │     │
│  │    + AdversarialPatch bypasses InputNormalization       │     │
│  │    + MyDroneDetector only has InputNormalization        │     │
│  │    → MyDroneDetector hasDefenseGap AdversarialPatch     │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│                           │                                      │
│                           ▼                                      │
│  OUTPUT: Inferred Facts                                          │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ NEWLY DERIVED (not explicitly stored):                  │     │
│  │ • AutoAttack targets YOLOv5                             │     │
│  │ • MyDroneDetector potentiallyVulnerableTo FGSM          │     │
│  │ • MyDroneDetector potentiallyVulnerableTo PGD           │     │
│  │ • MyDroneDetector potentiallyVulnerableTo AutoAttack    │     │
│  │ • MyDroneDetector hasDefenseGap AdversarialPatch        │     │
│  │ • CRITICAL ALERT: Physical attack vulnerability with    │     │
│  │   no deployed mitigation in Defense sector deployment   │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

# Part 4: Defence Case Study - Adversarial Camouflage

## 4.1 The Scenario

The Australian Defence Force deploys AI-powered drone surveillance to detect military vehicles. An adversary wants to hide their tanks from detection.

**Traditional camouflage:** Paint tanks to blend with environment (fools human eyes)

**Adversarial camouflage:** Apply specially computed patterns that fool AI detectors (YOLO, Faster R-CNN) even when clearly visible to humans

### Real Research Examples:

1. **Adversarial Patches (TNO, Netherlands):** A printed pattern placed on a military aircraft makes YOLO completely miss it, even though humans clearly see a plane.

2. **Dual Attribute Adversarial Camouflage (DAAC):** Patterns that fool BOTH AI and humans by matching scene features.

3. **CamoNet:** Target-specific perturbations for remote sensing that are imperceptible but cause detection failure.

## 4.2 How DARK-G Models This Threat

### Knowledge Graph Population

```turtle
# Attack Entities
aml:AdversarialPatch rdf:type aml:PhysicalEvasionAttack ;
    aml:targets aml:ObjectDetector ;
    aml:publishedBy "Brown et al." ;
    aml:publishedYear 2017 ;
    aml:physicalRealizability "high" ;
    aml:typicalPatchSize "10-30% of object" .

aml:DAAC rdf:type aml:PhysicalEvasionAttack ;
    aml:variantOf aml:AdversarialPatch ;
    aml:targets aml:ObjectDetector ;
    aml:foolsHumans "true"^^xsd:boolean ;
    aml:environmentAdaptive "true"^^xsd:boolean .

aml:CamoNet rdf:type aml:PhysicalEvasionAttack ;
    aml:targets aml:RemoteSensingDetector ;
    aml:perturbationVisibility "imperceptible" ;
    aml:applicationDomain aml:RemoteSensing .

# Model Entities
aml:YOLOv5 rdf:type aml:ObjectDetector ;
    aml:architecture aml:CNN ;
    aml:typicalUse "real-time detection" .

aml:DroneReconSystem rdf:type aml:DeployedModel ;
    aml:usesArchitecture aml:YOLOv5 ;
    aml:deployedIn aml:DefenseSector ;
    aml:operationalEnvironment aml:AerialSurveillance ;
    aml:hasDefense aml:JPEGCompression ;
    aml:hasDefense aml:SpatialSmoothing .

# Defense Entities
aml:JPEGCompression rdf:type aml:InputPreprocessing ;
    aml:mitigates aml:DigitalAdversarialExamples ;
    aml:effectivenessVs aml:AdversarialPatch "low" ;
    aml:computationalCost "minimal" .

aml:AdversarialTraining rdf:type aml:ProactiveDefense ;
    aml:mitigates aml:AdversarialPatch ;
    aml:effectivenessVs aml:AdversarialPatch "medium" ;
    aml:mitigates aml:DAAC ;
    aml:effectivenessVs aml:DAAC "low" ;
    aml:computationalCost "high" .

aml:DetectorEnsemble rdf:type aml:ArchitecturalDefense ;
    aml:mitigates aml:AdversarialPatch ;
    aml:effectivenessVs aml:AdversarialPatch "high" ;
    aml:mitigates aml:DAAC ;
    aml:effectivenessVs aml:DAAC "medium" ;
    aml:computationalCost "high" .

# Context
aml:DefenseSector rdf:type aml:Sector ;
    aml:regulatedBy aml:DefenceSecurityPolicy ;
    aml:threatActorProfile aml:NationState .
```

## 4.3 Query Examples

### Query 1: "What physical attacks threaten our drone detection system?"

```sparql
PREFIX aml: <http://dark-g.org/ontology#>

SELECT ?attack ?severity ?physicalRealizability ?foolsHumans
WHERE {
  aml:DroneReconSystem aml:usesArchitecture ?arch .
  ?attack rdf:type aml:PhysicalEvasionAttack .
  ?attack aml:targets ?targetType .
  ?arch rdf:type ?targetType .
  
  OPTIONAL { ?attack aml:physicalRealizability ?physicalRealizability }
  OPTIONAL { ?attack aml:foolsHumans ?foolsHumans }
  
  # Compute severity based on realizability and detection difficulty
  BIND(IF(?foolsHumans = true, "CRITICAL", "HIGH") AS ?severity)
}
```

**Result:**
| Attack | Severity | Physical Realizability | Fools Humans |
|--------|----------|------------------------|--------------|
| AdversarialPatch | HIGH | high | false |
| DAAC | CRITICAL | high | true |
| CamoNet | HIGH | medium | false |

### Query 2: "What are our defense gaps against camouflage attacks?"

```sparql
PREFIX aml: <http://dark-g.org/ontology#>

SELECT ?attack ?currentDefense ?effectiveness ?recommendedDefense ?improvedEffectiveness
WHERE {
  # Find attacks we're vulnerable to
  aml:DroneReconSystem aml:usesArchitecture ?arch .
  ?attack aml:targets ?arch .
  ?attack rdf:type aml:PhysicalEvasionAttack .
  
  # Find current defenses and their effectiveness
  OPTIONAL {
    aml:DroneReconSystem aml:hasDefense ?currentDefense .
    ?currentDefense aml:effectivenessVs ?attack ?effectiveness .
  }
  
  # Find better defenses we could deploy
  OPTIONAL {
    ?recommendedDefense aml:mitigates ?attack .
    ?recommendedDefense aml:effectivenessVs ?attack ?improvedEffectiveness .
    FILTER(?improvedEffectiveness > ?effectiveness || !BOUND(?effectiveness))
  }
}
```

**Result:**
| Attack | Current Defense | Effectiveness | Recommended Defense | Improved Effectiveness |
|--------|-----------------|---------------|--------------------|-----------------------|
| AdversarialPatch | JPEGCompression | low | DetectorEnsemble | high |
| AdversarialPatch | SpatialSmoothing | low | AdversarialTraining | medium |
| DAAC | JPEGCompression | none | DetectorEnsemble | medium |
| DAAC | SpatialSmoothing | none | AdversarialTraining | low |

### Query 3: "Alert when new camouflage attack is published"

When a new paper is ingested (e.g., "ShipCamou: Adaptive patches for maritime vessels"):

```sparql
# Continuous monitoring query
PREFIX aml: <http://dark-g.org/ontology#>

CONSTRUCT {
  ?org aml:newThreatAlert ?newAttack .
  ?newAttack aml:affectsSystem ?system .
  ?newAttack aml:urgency ?urgency .
}
WHERE {
  # New attack added in last 7 days
  ?newAttack aml:addedToGraph ?addDate .
  FILTER(?addDate > NOW() - "P7D"^^xsd:duration)
  
  # Check if it affects any registered systems
  ?system rdf:type aml:DeployedModel .
  ?system aml:usesArchitecture ?arch .
  ?newAttack aml:targets ?arch .
  
  # Check if system has defense
  OPTIONAL {
    ?system aml:hasDefense ?defense .
    ?defense aml:mitigates ?newAttack .
  }
  
  # Set urgency based on defense gap
  BIND(IF(!BOUND(?defense), "CRITICAL", "MEDIUM") AS ?urgency)
}
```

---

# Part 5: Why This Matters - The Transformative Impact

## 5.1 From Reactive to Proactive

| Aspect | Without DARK-G | With DARK-G |
|--------|----------------|-------------|
| **Threat Discovery** | Wait for incident or manual review | Automatic alert within 7 days of publication |
| **Vulnerability Assessment** | Manual expert analysis per model | Automated inference across all models |
| **Defense Selection** | Trial and error; vendor recommendations | Graph-driven, evidence-based recommendations |
| **Compliance** | Manual documentation | Auto-generated from graph queries |
| **Knowledge Sharing** | Siloed in teams | Unified, queryable, shareable |

## 5.2 Concrete Value Metrics

- **Time to threat awareness**: Months → Days
- **Manual assessment effort**: Weeks per model → Minutes per model
- **Defense coverage visibility**: Unknown → 100% mapped
- **Knowledge decay**: Rapid → Continuously updated
- **Reasoning capability**: None → Automated inference

## 5.3 Australian National Benefit

DARK-G provides Australia with:

1. **Sovereign capability** in AI security assessment
2. **Compliance tooling** for SOCI Act, Cyber Security Act, AI Plan requirements
3. **Defence advantage** through adversarial-aware AI procurement
4. **Export potential** to Five Eyes and allied nations
5. **Research leadership** in semantic AI security

---

# Summary

**The Problem:** Adversarial ML knowledge is fragmented, static, and cannot scale.

**The Solution:** A semantic knowledge graph that:
- **Represents** attacks, defenses, models, and context as interconnected entities
- **Reasons** about vulnerabilities through ontology rules and inference
- **Updates** continuously from research publications and threat intelligence
- **Queries** to answer complex questions about specific deployments
- **Simulates** attacks based on graph-derived threat prioritization

**The Key Innovation:** Moving from "storing facts" to "reasoning about knowledge" - using formal semantics (RDF, OWL, SPARQL) to derive new insights automatically.

**The Outcome:** Organisations can finally answer: "What threatens MY models, and what should I do about it?" - continuously, automatically, and with full traceability.
