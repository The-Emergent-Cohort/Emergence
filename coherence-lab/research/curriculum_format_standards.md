# Educational Curriculum Format Standards Research

*Research conducted December 10, 2025*

## Executive Summary

This document surveys human-created educational standards for structuring curriculum content. The goal is to adopt established formats rather than inventing our own, ensuring the DI training system is grounded in real pedagogy.

**Recommendation:** Use CASE/OpenSALT as the competency framework base, with LRMI for metadata, and custom extensions for CPA stages and gradual release pedagogy.

---

## 1. IMS Global / 1EdTech Standards

### CASE (Competencies and Academic Standards Exchange)

**What it covers:**
- Machine-readable competency frameworks and learning standards
- Hierarchical relationships between learning objectives
- Standards alignment across different frameworks
- URIs for universal identification of competencies

**Key Features:**
- JSON-LD based with REST API specification
- Supports prerequisite relationships and competency associations
- Designed to replace PDF/HTML standards documents with structured data
- OpenSALT is the reference implementation

**Data Model Objects:**
- `CFDocument` - Root container for competency frameworks
- `CFItem` - Individual competency or standard statement
- `CFAssociation` - Relationships between items (prerequisites, hierarchies)
- `CFRubric` - Assessment rubrics
- `CFSubject` - Subject area categorization

**Documentation:**
- https://www.imsglobal.org/activity/case
- https://purl.imsglobal.org/spec/case/v1p0/schema/json/

### QTI 3.0 (Question and Test Interoperability)

**What it covers:**
- Assessment items (questions) and tests
- Item metadata (difficulty, learning objectives)
- Scoring rules and response processing
- Adaptive testing

**Key Features:**
- XML-based with content packaging
- Native accessibility support (WCAG 2.1 AA)
- IEEE LOM metadata binding

**Documentation:**
- https://www.imsglobal.org/spec/qti/v3p0/oview

### Common Cartridge

**What it covers:**
- Complete digital course packages (textbooks + assessments + activities)
- Learning object packaging and distribution
- Curriculum standards alignment metadata

**Key Features:**
- XML manifest (imsmanifest.xml) with content packaging
- Integrates QTI for assessments, LTI for tools
- CASE URI support for standards alignment

### Caliper Analytics

**What it covers:**
- Learning activity event tracking
- Actor-Action-Object event model
- Real-time learner behavior data

**Key Features:**
- JSON-LD format for events
- Similar to xAPI Actor-Verb-Object structure
- Event profiles for different domains (Assessment, Grading, etc.)

---

## 2. SCORM / xAPI / cmi5

### xAPI (Experience API)

**What it covers:**
- Universal learning activity tracking (online and offline)
- Rich activity data beyond traditional LMS

**Statement Structure:**
```json
{
  "actor": {"name": "Student", "mbox": "mailto:student@example.com"},
  "verb": {"id": "http://adlnet.gov/expapi/verbs/completed"},
  "object": {"id": "http://example.com/course/lesson-01"}
}
```

**Why it matters:** We can use xAPI-style statements to track DI progress through curriculum.

### cmi5

- Bridge between SCORM and xAPI
- Course structure with xAPI tracking
- Assignable Units (AUs) concept

---

## 3. Metadata Standards

### LRMI (Learning Resource Metadata Initiative)

**What it covers:**
- Educational metadata extensions for Schema.org
- Web-discoverable learning resources

**Key Properties:**
- `educationalAlignment` - Link to standards
- `educationalUse` - Purpose (assignment, activity, etc.)
- `learningResourceType` - Type of resource
- `typicalAgeRange` - Target age
- `timeRequired` - Duration
- `interactivityType` - Active/passive learning

**This is where we hang CPA and I Do/We Do/You Do metadata.**

### IEEE LOM (Learning Object Metadata)

**9 Categories:**
1. General (title, language, description)
2. Life Cycle (version, status, contributors)
3. Meta-Metadata (schema info)
4. Technical (format, size, requirements)
5. Educational (difficulty, context, age range)
6. Rights (cost, copyright)
7. Relation (to other resources)
8. Annotation (comments, reviews)
9. Classification (taxonomy, keywords)

---

## 4. Singapore Math / Eureka Math Structure

### Singapore Math CPA Framework

**Pedagogical Stages:**
1. **Concrete** - Physical manipulatives (base-10 blocks, fraction tiles)
2. **Pictorial** - Visual representations (bar models, number bonds)
3. **Abstract** - Mathematical symbols and notation

**Scope & Sequence Structure:**
- Topics organized by grade level
- Polya's 4-step problem solving embedded
- Mastery before advancement

### Eureka Math Structure

**Hierarchy:**
- Grade → Module → Topic → Lesson
- Each module: Topics A-J (variable), assessment days
- Example: Grade 1, Module 1 = 37 lesson + 6 assessment = 43 days

**Documentation:**
- https://www.singaporemath.com/pages/scopes-sequences
- https://tea.texas.gov/academics/instructional-materials/tea-available-materials/eurekamathteks-scopeandsequence.pdf

---

## 5. Gradual Release of Responsibility

### I Do / We Do / You Do Framework

**Phases:**
1. **I Do** (Focused Instruction) - Teacher models, students observe
2. **We Do** (Guided Instruction) - Teacher and students work together
3. **You Do Together** (Collaborative) - Students work in groups
4. **You Do Alone** (Independent) - Students work individually

**Key Insight:** Scaffolding decreases as student competence increases.

**Sources:**
- Fisher & Frey (2014) - Better Learning Through Structured Teaching
- Pearson & Gallagher (1983) - Original gradual release model

---

## 6. Common Core Standards Format

**Official Format:**
- XML representation with official URIs
- CEDS schema (Common Education Data Standards)
- Append "XML" to identifier URL: `http://www.corestandards.org/Math/Content/K/CC/A/1/XML`

**Third-Party:**
- Common Standards Project - JSON/API for all 50 states
- OpenSALT/CASE - Modern replacement

---

## 7. Adopted Format Approach

Based on this research, we adopt:

### Base Structure
**ChatGPT-proposed hierarchy** (grounded in real textbook structure):
```
Curriculum → Subject → Course → Unit → Lesson → Activity
```

### Standards Integration
- **CASE URIs** for competency alignment (`competency_refs`)
- **xAPI activity IDs** for tracking (`xapi_activity_id`)
- **LRMI properties** for educational metadata

### Custom Extensions
At each level, `extensions` object holds:
- `cpa_stage`: concrete | pictorial | abstract
- `instructional_phase`: i_do | we_do | you_do
- `scaffolding_level`: 0.0 (none) to 1.0 (full)
- `native_representation`: subject-specific format
- `interleave_with`: cross-topic mixing for retention

---

## 8. Key Sources

### IMS Global / 1EdTech
- Common Cartridge: https://www.1edtech.org/standards/cc
- CASE Specification: https://www.imsglobal.org/activity/case
- QTI 3.0: https://www.imsglobal.org/spec/qti/v3p0/oview
- Caliper Analytics: https://www.imsglobal.org/spec/caliper/v1p2

### Metadata
- LRMI at DCMI: https://dublincore.org/about/lrmi/
- Schema.org LearningResource: https://schema.org/LearningResource
- IEEE LOM: https://standards.ieee.org/ieee/1484.12.1/7699/

### Pedagogy
- Singapore Math CPA: https://mathsnoproblem.com/en/approach/concrete-pictorial-abstract
- Gradual Release: Fisher & Frey (2014)

---

*This research informs the curriculum format specification in `curriculum_format_spec.md`*
